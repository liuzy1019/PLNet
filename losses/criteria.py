"""Training criteria used by PLNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    """Binary soft-Dice loss operating on probabilities."""

    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        if predict.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes must match: {predict.shape} != {target.shape}"
            )
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth

        dice_score = 2 * num / den
        loss_avg = 1 - dice_score.mean()

        return loss_avg


class DiceLoss(nn.Module):
    """Mean per-class Dice loss for multi-channel logits."""

    def __init__(self, weight=None, ignore_index=None, smooth=1):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, predict, target):
        if predict.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes must match: {predict.shape} != {target.shape}"
            )
        if self.ignore_index is not None and not 0 <= self.ignore_index < target.shape[1]:
            raise ValueError(f"ignore_index is outside the class range: {self.ignore_index}")

        dice = BinaryDiceLoss(smooth=self.smooth)
        total_loss = 0
        predict = torch.sigmoid(predict.float())
        target = target.float()

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    if len(self.weight) != target.shape[1]:
                        raise ValueError(
                            f"Expected {target.shape[1]} class weights, got {len(self.weight)}"
                        )
                    dice_loss *= self.weight[i]
                total_loss += dice_loss

        return total_loss / (
            target.shape[1] - 1 if self.ignore_index is not None else target.shape[1]
        )


class BCELoss(nn.Module):
    """Mean per-class binary cross-entropy for multi-channel logits."""

    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target):
        if predict.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes must match: {predict.shape} != {target.shape}"
            )
        if self.ignore_index is not None and not 0 <= self.ignore_index < target.shape[1]:
            raise ValueError(f"ignore_index is outside the class range: {self.ignore_index}")

        total_loss = 0
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                bce_loss = self.criterion(predict[:, i], target[:, i])
                total_loss += bce_loss

        return total_loss.mean()


class AuxiliaryBCELoss(nn.Module):
    """Class-balanced BCE for logits from the deep-supervision heads."""

    @staticmethod
    def weighted_bce_with_logits(logits, target, foreground_weight):
        logits = logits.float()
        target = target.float()
        positive_term = foreground_weight * target * F.logsigmoid(logits)
        negative_term = (1 - target) * F.logsigmoid(-logits)
        return -(positive_term + negative_term).mean()

    def forward(self, predict, target):
        if predict.shape != target.shape:
            raise ValueError(
                f"Prediction and target shapes must match: {predict.shape} != {target.shape}"
            )

        bce_loss = []
        for i in range(predict.shape[1]):
            pred_i = predict[:, i]
            targ_i = target[:, i]
            foreground = targ_i.float().sum()
            total_elements = torch.tensor(
                targ_i.numel(),
                dtype=torch.float32,
                device=pred_i.device,
            )
            foreground_weight = torch.log(total_elements / (foreground + 1)).clamp_min(0)
            bce_i = self.weighted_bce_with_logits(
                pred_i,
                targ_i,
                foreground_weight,
            )
            bce_loss.append(bce_i)

        return torch.stack(bce_loss).mean()
