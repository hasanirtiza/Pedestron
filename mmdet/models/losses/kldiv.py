import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import LOSSES
from .utils import weight_reduce_loss


def kl_divergence(source, target, reduction='mean'):
    
    return F.kl_div(source, target, reduction=reduction)

@LOSSES.register_module
class KLDivLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):
        super(KLDivLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.criterion = kl_divergence

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        cls_score = F.sigmoid(cls_score)
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.criterion(
            cls_score,
            label,
            reduction=reduction)
        return loss_cls
