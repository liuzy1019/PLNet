"""Evaluate a PLNet checkpoint and save MM-WHS NIfTI predictions."""

import argparse
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from datasets.mmwhs import CLASS_NAMES, LABEL_VALUES, HeartValDataset
from models import PLNet
from utils.config import load_config
from utils.inference import sliding_window_inference
from utils.metrics import compute_overlap_metrics, prediction_to_class_map


PROJECT_ROOT = Path(__file__).resolve().parent


def _parse_input_size(value):
    try:
        size = tuple(int(item) for item in value.split(","))
    except (AttributeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "input_size must be formatted as depth,height,width"
        ) from error
    if len(size) != 3 or any(item < 24 or item % 8 for item in size):
        raise argparse.ArgumentTypeError(
            "input_size must contain three multiples of 8, each at least 24"
        )
    return size


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")


def get_arguments():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str)
    known, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="PLNet evaluation for MM-WHS2017.",
        parents=[config_parser],
    )
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument(
        "--data_list",
        type=str,
        default="configs/data/manifests/mmwhs_ct.json",
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--input_size", type=str, default="80,160,160")
    parser.add_argument("--num_classes", type=int, default=len(CLASS_NAMES))
    parser.add_argument("--weight_std", type=_str_to_bool, default=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs/predictions")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overlap", type=float, default=0.7)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--raw_labels",
        action="store_true",
        help="Save original MM-WHS label values instead of class indices 0-7.",
    )

    if known.config:
        parser.set_defaults(**load_config(known.config))
    args = parser.parse_args()

    if args.num_classes != len(CLASS_NAMES):
        parser.error(f"MM-WHS requires {len(CLASS_NAMES)} output classes")
    if not args.checkpoint:
        parser.error("--checkpoint is required")
    if not 0 <= args.overlap < 1:
        parser.error("--overlap must satisfy 0 <= overlap < 1")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    args.input_size = _parse_input_size(args.input_size)

    for field in ("data_dir", "data_list", "checkpoint", "output_dir"):
        value = Path(getattr(args, field)).expanduser()
        if not value.is_absolute():
            value = PROJECT_ROOT / value
        setattr(args, field, value.resolve())
    return args


def load_checkpoint(model, checkpoint_path):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must contain a model state dictionary")
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)


def save_prediction(class_map, sample, output_dir, raw_labels):
    prediction = class_map.detach().cpu().numpy().astype(np.uint8)
    prediction = prediction.transpose(1, 2, 0)
    if raw_labels:
        prediction = np.asarray(LABEL_VALUES, dtype=np.int16)[prediction]
    output_path = output_dir / f"{sample['image_id']}_label.nii.gz"
    nifti = nib.Nifti1Image(
        prediction,
        affine=sample["affine"],
        header=sample["header"],
    )
    nifti.set_data_dtype(prediction.dtype)
    nib.save(nifti, output_path)
    return output_path


def main():
    args = get_arguments()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)

    model = PLNet(
        args.input_size,
        num_classes=args.num_classes,
        weight_std=args.weight_std,
    )
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    dataset = HeartValDataset(
        args.data_dir,
        args.data_list,
        subset=args.split,
        return_metadata=True,
    )
    if not dataset:
        raise ValueError(f"The {args.split} split is empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_dice = []
    all_iou = []
    total_time = 0.0
    for index in range(len(dataset)):
        sample = dataset[index]
        image = torch.from_numpy(sample["image"]).unsqueeze(0).to(device)
        target = torch.from_numpy(sample["label"]).unsqueeze(0)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            probabilities = sliding_window_inference(
                model,
                image,
                args.input_size,
                args.num_classes,
                overlap=args.overlap,
                apply_sigmoid=True,
                accumulator_device="cpu",
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        dice, iou = compute_overlap_metrics(
            probabilities,
            target,
            args.threshold,
            from_logits=False,
        )
        all_dice.append(dice)
        all_iou.append(iou)
        class_map = prediction_to_class_map(
            probabilities,
            args.threshold,
            from_logits=False,
        )[0]
        output_path = save_prediction(
            class_map,
            sample,
            args.output_dir,
            args.raw_labels,
        )
        print(
            f"[{index + 1}/{len(dataset)}] {sample['image_id']}: "
            f"Dice={np.mean(dice):.4f}, time={elapsed:.2f}s, saved={output_path}"
        )

    mean_dice = np.asarray(all_dice).mean(axis=0)
    mean_iou = np.asarray(all_iou).mean(axis=0)
    labels = CLASS_NAMES[1:]
    print("Mean Dice:", " | ".join(f"{n}={v:.4f}" for n, v in zip(labels, mean_dice)))
    print("Mean IoU:", " | ".join(f"{n}={v:.4f}" for n, v in zip(labels, mean_iou)))
    print(f"Average inference time: {total_time / len(dataset):.2f}s")


if __name__ == "__main__":
    main()
