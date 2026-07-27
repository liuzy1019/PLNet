import argparse
import os
import os.path as osp
import random
import time
import timeit
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from datasets.mmwhs import CLASS_NAMES, HeartDataset, HeartValDataset
from losses import AuxiliaryBCELoss, BCELoss, DiceLoss
from models import PLNet
from utils.config import load_config
from utils.inference import sliding_window_inference
from utils.metrics import compute_overlap_metrics

PROJECT_ROOT = Path(__file__).resolve().parent


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def lr_poly(base_lr, current_iter, max_iter, power):
    return base_lr * ((1 - float(current_iter) / max_iter) ** power)


def adjust_learning_rate(optimizer, i_iter, lr, num_steps, power):
    lr = lr_poly(lr, i_iter, num_steps, power)
    optimizer.param_groups[0]["lr"] = lr

    return lr


def compute_training_loss(preds, labels, dice_loss, bce_loss, auxiliary_loss):
    """Combine main Dice/BCE losses with weighted deep-supervision BCE."""
    outputs = list(preds) if isinstance(preds, (list, tuple)) else [preds]
    if len(outputs) != 4:
        raise ValueError(
            f"PLNet must return one main and three auxiliary outputs, got {len(outputs)}"
        )
    loss = dice_loss(outputs[0], labels) + bce_loss(outputs[0], labels)

    deep_supervision_weights = (0.5, 0.25, 0.25)
    for weight, output in zip(deep_supervision_weights, outputs[1:]):
        loss = loss + weight * auxiliary_loss(output, labels)
    return loss


def validate(input_size, model, val_loader, num_classes, dice_loss, bce_loss, device):
    """Evaluate full validation volumes with overlapping sliding windows."""
    model.eval()
    foreground_classes = num_classes - 1
    val_dice = [0.0] * foreground_classes
    val_iou = [0.0] * foreground_classes
    num = len(val_loader)
    if num == 0:
        raise ValueError("Validation loader is empty.")
    for idx, batch in enumerate(val_loader):
        image, label = batch
        image, label = image.to(device), label.to(device)

        with torch.no_grad():
            pred = sliding_window_inference(
                model,
                image,
                input_size,
                num_classes,
                overlap=1 / 3,
            )
            del image

            val_loss = dice_loss(pred, label) + bce_loss(pred, label)

            dice, iou = compute_overlap_metrics(pred, label)
            del pred, label

            for i in range(len(dice)):
                val_dice[i] += dice[i]
                val_iou[i] += iou[i]

            mean_dice = np.mean(dice)
            mean_iou = np.mean(iou)
            dice_by_class = " | ".join(
                f"{CLASS_NAMES[class_id]}={score:.5f}"
                for class_id, score in enumerate(dice, start=1)
            )
            iou_by_class = " | ".join(
                f"{CLASS_NAMES[class_id]}={score:.5f}"
                for class_id, score in enumerate(iou, start=1)
            )
            print(
                f"Validate: sample={idx}, loss={val_loss:.5f}, "
                f"mean_dice={mean_dice:.5f} | {dice_by_class}"
            )
            print(
                f"Validate: sample={idx}, loss={val_loss:.5f}, "
                f"mean_iou={mean_iou:.5f} | {iou_by_class}"
            )

    avg_val_dice = [i / (num) for i in val_dice]
    avg_val_iou = [i / (num) for i in val_iou]

    return avg_val_dice, avg_val_iou


def get_arguments():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    known, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Train PLNet for 3D medical image segmentation.",
        parents=[config_parser],
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(PROJECT_ROOT),
        help="Repository root containing the data directory.",
    )
    parser.add_argument("--data_mod", choices=("ct", "mr"), default="ct")
    parser.add_argument(
        "--train_list",
        type=str,
        default=str(PROJECT_ROOT / "configs/data/manifests/mmwhs_ct.json"),
    )
    parser.add_argument(
        "--val_list",
        type=str,
        default=str(PROJECT_ROOT / "configs/data/manifests/mmwhs_ct.json"),
    )
    parser.add_argument(
        "--input_size",
        type=str,
        default="80,160,160",
        help="Comma-separated depth,height,width crop size.",
    )
    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--model", type=str, default="plnet")
    parser.add_argument(
        "--snapshot_dir", type=str, default=str(PROJECT_ROOT / "outputs/checkpoints")
    )
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=False)
    parser.add_argument("--reload_path", type=str, default="")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--amp", default=0, type=int, choices=(0, 1), help="Enable automatic mixed precision."
    )
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--start_iters", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=40000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--random_mirror", type=str2bool, default=True)
    parser.add_argument("--random_scale", type=str2bool, default=True)
    parser.add_argument("--valid_on_train", default=1, type=int, choices=(0, 1))
    parser.add_argument("--val_pred_every", type=int, default=5000)
    if known.config:
        parser.set_defaults(**load_config(known.config))
    args = parser.parse_args()
    if args.num_classes != len(CLASS_NAMES):
        parser.error(
            f"MM-WHS requires {len(CLASS_NAMES)} classes, including background; "
            f"received {args.num_classes}"
        )
    if not 0 <= args.start_iters < args.num_steps:
        parser.error("start_iters must satisfy 0 <= start_iters < num_steps")
    if args.batch_size <= 0:
        parser.error("batch_size must be positive")
    if args.num_workers < 0:
        parser.error("num_workers must be non-negative")
    if args.local_rank < 0:
        parser.error("local_rank must be non-negative")
    if args.learning_rate <= 0:
        parser.error("learning_rate must be positive")
    if args.weight_decay < 0:
        parser.error("weight_decay must be non-negative")
    if args.val_pred_every <= 0:
        parser.error("val_pred_every must be positive")
    try:
        input_size = tuple(int(value) for value in args.input_size.split(","))
    except (AttributeError, ValueError):
        parser.error("input_size must be formatted as depth,height,width")
    if len(input_size) != 3 or any(value < 24 or value % 8 for value in input_size):
        parser.error(
            "input_size must contain three multiples of 8, each at least 24"
        )
    for field in ("data_dir", "train_list", "val_list", "snapshot_dir", "reload_path"):
        value = getattr(args, field)
        if value and not Path(value).is_absolute():
            setattr(args, field, str((PROJECT_ROOT / value).resolve()))
    return args


if __name__ == "__main__":
    args = get_arguments()
    # 1) Runtime setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError("PLNet training requires a CUDA-capable GPU.")
    device = torch.device(f"cuda:{args.local_rank}")

    cudnn.benchmark = True
    cudnn.deterministic = False

    # Random seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    # Precision mode
    if args.amp == 0:
        print("Training with full precision")
    else:
        print("Training with automatic mixed precision")

    # 2) Datasets
    d, h, w = map(int, args.input_size.split(","))
    input_size = (d, h, w)

    db_train = HeartDataset(
        args.data_dir,
        args.train_list,
        max_iters=args.num_steps * args.batch_size,
        crop_size=input_size,
        scale=args.random_scale,
        mirror=args.random_mirror,
        subset="train",
    )
    if args.valid_on_train:
        db_val = HeartValDataset(args.data_dir, args.val_list, subset="val")

    print(f"\nSamples for train = {len(db_train)}")
    if args.valid_on_train:
        print(f"Samples for valid = {len(db_val)}")

    trainloader = DataLoader(
        db_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    valloader = (
        DataLoader(db_val, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
        if args.valid_on_train
        else None
    )

    # 3) Model
    model = PLNet(input_size, num_classes=args.num_classes, weight_std=args.weight_std)
    model.to(device)

    if args.reload_from_checkpoint:
        print(f"loading from checkpoint: {args.reload_path}")
        if not os.path.exists(args.reload_path):
            raise FileNotFoundError(f"Checkpoint does not exist: {args.reload_path}")
        state_dict = torch.load(
            args.reload_path,
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        model.load_state_dict(state_dict)

    # Run output directory
    start_datetime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    if args.model:
        args.snapshot_dir = args.snapshot_dir + "/" + args.model
    else:
        args.snapshot_dir = args.snapshot_dir

    save_pth = args.snapshot_dir + "/" + str(start_datetime) + "/" + args.data_mod
    os.makedirs(save_pth, exist_ok=False)
    print(save_pth)
    writer = SummaryWriter(save_pth)

    # 4) Training
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    loss_D = DiceLoss().to(device)
    loss_BCE = BCELoss().to(device)
    loss_B = AuxiliaryBCELoss().to(device)
    amp_grad_scaler = GradScaler(enabled=(args.amp != 0))
    total_loss = []
    val_dice = []
    val_iou = []
    start = timeit.default_timer()
    print("\n===============================Train===================================")
    for i_iter, sampled_batch in enumerate(trainloader, start=args.start_iters):
        images, labels = sampled_batch
        images, labels = images.to(device), labels.to(device)

        model.train()
        optimizer.zero_grad()

        # Forward and optimization
        if args.amp == 0:
            preds = model(images)
            loss = compute_training_loss(preds, labels, loss_D, loss_BCE, loss_B)
            loss.backward()
            optimizer.step()

        else:
            with autocast(device_type="cuda"):
                preds = model(images)
                loss = compute_training_loss(preds, labels, loss_D, loss_BCE, loss_B)

            amp_grad_scaler.scale(loss).backward()
            amp_grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()

        # Polynomial learning-rate schedule
        lr = adjust_learning_rate(optimizer, i_iter, args.learning_rate, args.num_steps, args.power)

        # Training log
        total_loss.append(loss.item())
        print(f"Iteration = {i_iter}, lr = {lr:.6f}, loss = {loss.detach().cpu().numpy():.6f}")

        if i_iter % 200 == 0 and (args.local_rank == 0):
            writer.add_scalar("loss", loss.detach().cpu().numpy(), i_iter)

        # Validation
        if args.valid_on_train == 1:
            if i_iter % args.val_pred_every == 0 and i_iter != 0:
                print("===============================Validate===================================")
                dice, iou = validate(
                    input_size,
                    model,
                    valloader,
                    args.num_classes,
                    loss_D,
                    loss_BCE,
                    device,
                )

                mean_dice = np.mean(dice)
                mean_iou = np.mean(iou)
                val_dice.append(mean_dice)
                val_iou.append(mean_iou)

                for class_id, class_dice in enumerate(dice, start=1):
                    writer.add_scalar(f"Val_Class{class_id}_Dice", class_dice, i_iter)
                writer.add_scalar("Val_Mean_Dice", mean_dice, i_iter)
                writer.add_scalar("Val_Mean_Iou", mean_iou, i_iter)

                print(
                    f"Validate: iteration={i_iter}, mean_dice={mean_dice:.5f}, "
                    f"mean_iou={mean_iou:.5f}"
                )

                print("=============================Validate end=================================")

        # Checkpoint
        if i_iter % args.val_pred_every == 0 and i_iter != 0 and (args.local_rank == 0):
            print("Save model ...")
            if args.valid_on_train == 1:
                file_name = osp.join(
                    save_pth, "model_iter{}_{}.pth".format(i_iter, format(mean_dice, ".4f"))
                )
            else:
                file_name = osp.join(save_pth, f"model_iter{i_iter}.pth")
            torch.save(model.state_dict(), file_name)

        if i_iter >= args.num_steps - 1 and (args.local_rank == 0):
            print("Save final model ...")
            file_name = osp.join(save_pth, "last_model.pth")
            torch.save(model.state_dict(), file_name)
            break

    end = timeit.default_timer()
    total_time = (end - start) / 3600
    print(f"The total training time is {total_time:.2f} hours")
    print(
        "================================= Training process finished! ================================="
    )

    # 5) Training curves
    fig_dir = save_pth
    plt.figure(figsize=(8, 6))
    plt.plot(
        range(0, len(total_loss), 500),
        total_loss[::500],
        marker=".",
        ls="-",
        alpha=0.8,
        label="Total loss",
    )
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Training Loss", fontsize=14)
    plt.legend(fontsize=14, loc="upper right")
    plt.savefig(os.path.join(fig_dir, "train_loss.png"))
    plt.close()

    if val_dice:
        val_interval = args.val_pred_every
        x = [i * val_interval for i in range(1, len(val_dice) + 1)]
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(x, val_dice, "b-", marker=".", label="Mean DSC")
        ax1.set_xlabel("Iteration", fontsize=14)
        ax1.set_ylabel("DSC", color="b", fontsize=16)
        ax1.tick_params(axis="y", labelcolor="k")

        ax2 = ax1.twinx()
        ax2.plot(x, val_iou, "r--", marker=".", label="Mean IoU")
        ax2.set_ylabel("IoU", color="r", fontsize=16)
        ax2.tick_params(axis="y", labelcolor="k")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "val_metrics.png"))
        plt.close(fig)

    writer.close()
