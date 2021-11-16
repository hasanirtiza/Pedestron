import torch.nn as nn

from .bbox_head import BBoxHead
from ..registry import HEADS
from ..utils import ConvModule
import torch.nn.functional as F
from ..losses import accuracy
import torch
from mmdet.core import (delta2bbox, multiclass_nms, bbox_target, force_fp32,
                        auto_fp16)


@HEADS.register_module
class RefineHead(BBoxHead):
    """More general bbox head, with dropouts.
    """

    def __init__(self,
                 num_cls_convs=2,
                 num_cls_fcs=2,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 alpha=0.5,
                 weight_decay=0.0005,
                 *args,
                 **kwargs):
        super(RefineHead, self).__init__(*args, num_classes=1, with_reg=False, **kwargs)
        assert (num_cls_convs + num_cls_fcs > 0)

        self.alpha = alpha
        self.reg_lambda = weight_decay
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.in_channels)

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        self.fc_cls = nn.Linear(self.cls_last_dim, 1)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:

            last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
                branch_fcs.append(nn.Dropout(0.5))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(RefineHead, self).init_weights()
        for module_list in [self.cls_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def loss(self,
             cls_score,
             labels,
             label_weights,
             reduction_override=None, **kwargs):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            reg_loss = torch.tensor(0.0).float().cuda()
            for param in self.parameters():
                reg_loss += param.square().sum()
            losses['loss_cls'] += self.reg_lambda * reg_loss
            losses['acc'] = accuracy(cls_score, labels)
        return losses

    def forward(self, x):

        _cls = x
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        cls_score = self.fc_cls(x_cls)
        return cls_score

    def get_scores(self, x):

        cls_score = self.forward(x)
        return F.sigmoid(cls_score)

    def combine_scores(self, results, scores):
        results = results[0]
        results[:, 4] *= self.half()
        results[:, 4] += (1.0 - self.alpha) * scores
        return [results.cpu().numpy()]

