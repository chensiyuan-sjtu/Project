"""Two-view Dice + IoU supervision and projected invariant/sensitive losses."""
from __future__ import annotations

import torch


def dice_iou_loss(pred, target, eps=1e-6):
    axes = (1, 2, 3)
    inter = (pred * target).sum(axes)
    total = pred.sum(axes) + target.sum(axes)
    dice = 1 - (2 * inter + eps) / (total + eps)
    iou = 1 - (inter + eps) / (total - inter + eps)
    return (dice + iou).mean()


def detection_loss(pred_o, pred_a, target):
    return dice_iou_loss(pred_o, target) + dice_iou_loss(pred_a, target)


def invariant_alignment_loss(inv_o, inv_a):
    return torch.stack([(1 - (a * b).sum(1)).mean() for a, b in zip(inv_o, inv_a)]).mean()


def orthogonality_loss(inv_o, sen_o, inv_a, sen_a, enabled=True):
    if not enabled:
        return inv_o[0].new_zeros(())
    terms = [(a * b).sum(1).abs().mean()
             for inv, sen in [(inv_o, sen_o), (inv_a, sen_a)]
             for a, b in zip(inv, sen)]
    return torch.stack(terms).mean()


def build_loss_dict(outputs, target, cfg):
    original, augmented = outputs["original"], outputs["augmented"]
    det = detection_loss(original["pred"], augmented["pred"], target)
    ia = invariant_alignment_loss(original["inv_desc"], augmented["inv_desc"])
    orth = orthogonality_loss(original["inv_desc"], original["sen_desc"],
                              augmented["inv_desc"], augmented["sen_desc"],
                              cfg["model"].get("use_dual_stream", True))
    weights = cfg["loss"]
    regularization = weights.get("lambda_ia", 1.0) * ia + weights.get("lambda_orth", 1.0) * orth
    total = det + weights.get("lambda_reg", 0.1) * regularization
    return {"loss_det": det, "loss_ia": ia, "loss_orth": orth, "loss_total": total}
