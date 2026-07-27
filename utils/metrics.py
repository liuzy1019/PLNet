"""Overlap metrics for MM-WHS segmentation."""

import torch


def _dice_score(prediction, target):
    prediction = prediction.reshape(prediction.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = (prediction * target).sum(dim=1)
    denominator = prediction.sum(dim=1) + target.sum(dim=1)
    score = torch.where(
        denominator > 0,
        2 * intersection / denominator,
        torch.ones_like(denominator),
    )
    return score.mean()


def _iou_score(prediction, target):
    prediction = prediction.reshape(prediction.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    intersection = (prediction * target).sum(dim=1)
    union = (prediction + target - prediction * target).sum(dim=1)
    score = torch.where(
        union > 0,
        intersection / union,
        torch.ones_like(union),
    )
    return score.mean()


def compute_overlap_metrics(prediction, target, threshold=0.5, from_logits=True):
    """Return foreground Dice and IoU using the paper's sigmoid threshold."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes must match: {prediction.shape} != {target.shape}"
        )
    if not 0 <= threshold <= 1:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

    if from_logits:
        prediction = torch.sigmoid(prediction)
    prediction = (prediction >= threshold).to(target.dtype)
    dice = []
    iou = []
    for class_index in range(1, target.shape[1]):
        dice.append(
            _dice_score(prediction[:, class_index], target[:, class_index])
            .detach()
            .cpu()
            .item()
        )
        iou.append(
            _iou_score(prediction[:, class_index], target[:, class_index])
            .detach()
            .cpu()
            .item()
        )
    return dice, iou


def prediction_to_class_map(prediction, threshold=0.5, from_logits=True):
    """Collapse thresholded channels exactly as in the original evaluation code."""
    if from_logits:
        prediction = torch.sigmoid(prediction)
    prediction = prediction >= threshold
    class_map = torch.zeros(
        prediction.shape[0],
        *prediction.shape[2:],
        dtype=torch.long,
        device=prediction.device,
    )
    for class_index in range(prediction.shape[1]):
        class_map[prediction[:, class_index]] = class_index
    return class_map
