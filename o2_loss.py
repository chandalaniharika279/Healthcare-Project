import torch
import torch.nn.functional as F

def asymmetric_bce_loss(y_pred, y_true, fn_weight=10.0, fp_weight=1.0):
    """
    y_true: 0 or 1
    y_pred: probability [0,1]
    """
    bce = F.binary_cross_entropy(y_pred, y_true, reduction="none")

    fn_mask = (y_true == 1).float()   # emergency missed
    fp_mask = (y_true == 0).float()   # false alarm

    loss = fn_weight * fn_mask * bce + fp_weight * fp_mask * bce
    return loss.mean()
