import torch

from mmdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class MaskCost(object):
    """MaskCost.

    Args:
        weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1., pred_act=False, act_mode='sigmoid'):
        self.weight = weight
        self.pred_act = pred_act
        self.act_mode = act_mode

    def __call__(self, pred_instances, gt_instances, img_meta=None):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        cls_pred = pred_instances.masks
        target = gt_instances.masks
        if self.pred_act and self.act_mode == 'sigmoid':
            cls_pred = cls_pred.sigmoid()
        elif self.pred_act:
            cls_pred = cls_pred.softmax(dim=0)

        _, H, W = target.shape
        # flatten_cls_pred = cls_pred.view(num_proposals, -1)
        # eingum is ~10 times faster than matmul
        pos_cost = torch.einsum('nhw,mhw->nm', cls_pred, target)
        neg_cost = torch.einsum('nhw,mhw->nm', 1 - cls_pred, 1 - target)
        cls_cost = -(pos_cost + neg_cost) / (H * W)
        return cls_cost * self.weight
