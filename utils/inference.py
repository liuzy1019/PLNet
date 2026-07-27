"""Sliding-window inference for 3D volumes."""

from math import ceil

import torch
import torch.nn.functional as F


def sliding_window_inference(
    model,
    image,
    tile_size,
    num_classes,
    overlap=1 / 3,
    apply_sigmoid=False,
    accumulator_device=None,
):
    """Average overlapping PLNet outputs over a full volume."""
    if image.ndim != 5:
        raise ValueError(f"Expected a 5D image tensor, got {tuple(image.shape)}")
    if len(tile_size) != 3 or any(size <= 0 for size in tile_size):
        raise ValueError(f"tile_size must contain three positive values: {tile_size}")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must satisfy 0 <= overlap < 1, got {overlap}")

    original_size = tuple(image.shape[2:])
    pad_d = max(tile_size[0] - original_size[0], 0)
    pad_h = max(tile_size[1] - original_size[1], 0)
    pad_w = max(tile_size[2] - original_size[2], 0)
    if pad_d or pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h, 0, pad_d))

    _, _, depth, height, width = image.shape
    stride_d = max(1, ceil(tile_size[0] * (1 - overlap)))
    stride_h = max(1, ceil(tile_size[1] * (1 - overlap)))
    stride_w = max(1, ceil(tile_size[2] * (1 - overlap)))
    tile_depths = ceil((depth - tile_size[0]) / stride_d) + 1
    tile_rows = ceil((height - tile_size[1]) / stride_h) + 1
    tile_cols = ceil((width - tile_size[2]) / stride_w) + 1

    if accumulator_device is None:
        accumulator_device = image.device
    accumulator_device = torch.device(accumulator_device)

    output_sum = torch.zeros(
        (image.shape[0], num_classes, depth, height, width),
        dtype=torch.float32,
        device=accumulator_device,
    )
    counts = torch.zeros(
        (image.shape[0], 1, depth, height, width),
        dtype=torch.float32,
        device=accumulator_device,
    )

    for dep in range(tile_depths):
        for row in range(tile_rows):
            for col in range(tile_cols):
                d2 = min(dep * stride_d + tile_size[0], depth)
                h2 = min(row * stride_h + tile_size[1], height)
                w2 = min(col * stride_w + tile_size[2], width)
                d1 = d2 - tile_size[0]
                h1 = h2 - tile_size[1]
                w1 = w2 - tile_size[2]

                tile = image[:, :, d1:d2, h1:h2, w1:w2]
                outputs = model(tile)
                tile_logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                expected_shape = (
                    image.shape[0],
                    num_classes,
                    tile_size[0],
                    tile_size[1],
                    tile_size[2],
                )
                if tuple(tile_logits.shape) != expected_shape:
                    raise ValueError(
                        f"Model output has shape {tuple(tile_logits.shape)}, "
                        f"expected {expected_shape}"
                    )

                tile_output = torch.sigmoid(tile_logits) if apply_sigmoid else tile_logits
                tile_output = tile_output.detach().to(
                    device=accumulator_device,
                    dtype=torch.float32,
                )
                output_sum[:, :, d1:d2, h1:h2, w1:w2] += tile_output
                counts[:, :, d1:d2, h1:h2, w1:w2] += 1

    output = output_sum / counts
    return output[
        :,
        :,
        : original_size[0],
        : original_size[1],
        : original_size[2],
    ]
