import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for multi-class segmentation"""
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight)
        self.smooth = 1e-5

    def forward(self, predict, target):
        N, C = predict.size()[:2]
        predict = predict.view(N, C, -1)
        target = target.view(N, 1, -1)
        predict = F.softmax(predict, dim=1)
        target_onehot = torch.zeros(predict.size(), device=predict.device, dtype=predict.dtype)
        target_onehot.scatter_(1, target.long(), 1)
        intersection = torch.sum(predict * target_onehot, dim=2)
        union = torch.sum(predict.pow(2), dim=2) + torch.sum(target_onehot, dim=2)
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)
        if hasattr(self, 'weight'):
            if self.weight.type() != predict.type():
                self.weight = self.weight.type_as(predict)
            dice_coef = dice_coef * self.weight * C
        dice_loss = 1 - torch.mean(dice_coef)
        return dice_loss


class MultiClassFocalLoss(nn.Module):
    """Multi-class Focal Loss implementation"""
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.dim() == 4:
            N, C, H, W = inputs.shape
            inputs = inputs.view(N, C, -1)
            if targets.dim() == 4:
                targets = targets.view(N, C, -1)
            elif targets.dim() == 3:
                targets = targets.view(N, -1)
                targets_one_hot = torch.zeros(N, C, H * W, device=inputs.device, dtype=inputs.dtype)
                targets_one_hot.scatter_(1, targets.unsqueeze(1).long(), 1)
                targets = targets_one_hot
        else:
            if targets.dim() == 1:
                N, C = inputs.shape
                targets_one_hot = torch.zeros(N, C, device=inputs.device, dtype=inputs.dtype)
                targets_one_hot.scatter_(1, targets.unsqueeze(1).long(), 1)
                targets = targets_one_hot

        probs = F.softmax(inputs, dim=1)
        p_t = (probs * targets).sum(dim=1)
        focal_weight = (1 - p_t) ** self.gamma
        log_probs = F.log_softmax(inputs, dim=1)
        ce_loss = -(targets * log_probs).sum(dim=1)

        if self.alpha is not None:
            if self.alpha.dim() == 1:
                alpha_t = (self.alpha.unsqueeze(0).unsqueeze(-1) * targets).sum(dim=1)
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
