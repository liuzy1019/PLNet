import random

import nibabel as nib
import numpy as np
from skimage.transform import resize
from torch.utils import data

from .paths import load_manifest, resolve_data_path


LABELS = (
    (0, "background"),
    (205, "myocardium"),
    (420, "left_atrium"),
    (500, "left_ventricle"),
    (550, "right_atrium"),
    (600, "right_ventricle"),
    (820, "ascending_aorta"),
    (850, "pulmonary_artery"),
)
CLASS_NAMES = tuple(name for _, name in LABELS)
LABEL_VALUES = tuple(value for value, _ in LABELS)


def _encode_label(label):
    masks = [label == value for value in LABEL_VALUES]
    assigned = np.logical_or.reduce(masks)
    if not assigned.all():
        unknown_values = np.unique(label[~assigned])
        raise ValueError(f"Unsupported MM-WHS label values: {unknown_values.tolist()}")
    return np.stack(masks, axis=0).astype(np.float32)


def _random_flip(image, label):
    draw = np.random.random()
    if draw <= 0.3:
        return image, label

    choice = min(7, int((draw - 0.3) / 0.1) + 1)
    axes_by_choice = {
        1: (3,),
        2: (2,),
        3: (1,),
        4: (2, 3),
        5: (1, 3),
        6: (1, 2),
        7: (1, 2, 3),
    }
    for axis in axes_by_choice[choice]:
        image = np.flip(image, axis=axis)
        label = np.flip(label, axis=axis)
    return image, label


class HeartDataset(data.Dataset):
    """MM-WHS training dataset with random crop and flip augmentation."""

    def __init__(
        self,
        root,
        list_path,
        max_iters=None,
        crop_size=(80, 160, 160),
        scale=False,
        mirror=True,
        subset="train",
    ):
        if subset != "train":
            raise ValueError("HeartDataset only supports the training split.")

        self.root = root
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.mirror = mirror

        manifest = load_manifest(list_path)
        items = list(manifest["train"])
        if max_iters is not None:
            repeats = int(np.ceil(float(max_iters) / len(items)))
            items = (items * repeats)[:max_iters]

        self.files = [
            {
                "image": resolve_data_path(root, item["path"]),
                "label": resolve_data_path(root, item["label"]),
            }
            for item in items
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        item = self.files[index]
        image = nib.load(item["image"]).get_fdata().astype(np.float32)
        label = nib.load(item["label"]).get_fdata()
        if image.shape != label.shape or image.ndim != 3:
            raise ValueError(
                f"Image and label must be matching 3D volumes: {image.shape} != {label.shape}"
            )

        if self.scale:
            scale = np.random.uniform(0.9, 1.1)
        else:
            scale = 1.0
        crop_d = int(self.crop_d * scale)
        crop_h = int(self.crop_h * scale)
        crop_w = int(self.crop_w * scale)

        image_h, image_w, image_d = label.shape
        if image_d < crop_d or image_h < crop_h or image_w < crop_w:
            raise ValueError(
                f"Volume {label.shape} is smaller than requested crop "
                f"(height={crop_h}, width={crop_w}, depth={crop_d})"
            )
        d_offset = random.randint(0, max(image_d - crop_d, 0))
        h_offset = random.randint(0, max(image_h - crop_h, 0))
        w_offset = random.randint(0, max(image_w - crop_w, 0))

        image = image[
            h_offset : h_offset + crop_h,
            w_offset : w_offset + crop_w,
            d_offset : d_offset + crop_d,
        ]
        label = label[
            h_offset : h_offset + crop_h,
            w_offset : w_offset + crop_w,
            d_offset : d_offset + crop_d,
        ]

        image = image[None].transpose((0, 3, 1, 2))
        label = _encode_label(label).transpose((0, 3, 1, 2))

        if self.mirror:
            image, label = _random_flip(image, label)

        if self.scale:
            image = resize(
                image,
                (1, self.crop_d, self.crop_h, self.crop_w),
                order=1,
                mode="constant",
                cval=0,
                clip=True,
                preserve_range=True,
            )
            label = resize(
                label,
                (8, self.crop_d, self.crop_h, self.crop_w),
                order=0,
                mode="edge",
                cval=0,
                clip=True,
                preserve_range=True,
            )

        return image.astype(np.float32).copy(), label.astype(np.float32).copy()


class HeartValDataset(data.Dataset):
    """MM-WHS validation or test dataset."""

    def __init__(self, root, list_path, subset="val", return_metadata=False):
        if subset not in {"val", "test"}:
            raise ValueError("HeartValDataset supports only val and test splits.")

        self.return_metadata = return_metadata
        manifest = load_manifest(list_path)
        self.files = [
            {
                "image_id": item["image_id"],
                "image": resolve_data_path(root, item["path"]),
                "label": resolve_data_path(root, item["label"]),
            }
            for item in manifest[subset]
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        item = self.files[index]
        image_nifti = nib.load(item["image"])
        label_nifti = nib.load(item["label"])
        image = image_nifti.get_fdata().astype(np.float32)
        label = label_nifti.get_fdata()
        if image.shape != label.shape or image.ndim != 3:
            raise ValueError(
                f"Image and label must be matching 3D volumes: {image.shape} != {label.shape}"
            )

        image = image[None].transpose((0, 3, 1, 2))
        label = _encode_label(label).transpose((0, 3, 1, 2))
        image = image.astype(np.float32).copy()
        label = label.astype(np.float32).copy()
        if not self.return_metadata:
            return image, label
        return {
            "image": image,
            "label": label,
            "image_id": item["image_id"],
            "affine": image_nifti.affine.copy(),
            "header": image_nifti.header.copy(),
        }
