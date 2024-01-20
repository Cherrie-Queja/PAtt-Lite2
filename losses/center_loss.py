import torch.nn as nn
import torch

from losses.hard_mine_triplet_loss import TripletLoss


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=7, feat_dim=2, use_gpu=True, loss_weight=1.0):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.loss_weight = loss_weight

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels, **kwargs):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        # print(f'x shape: {x.shape}')
        # print(f'labels shape: {labels.shape}')
        batch_size = x.size(0)
        labels_=torch.Tensor(batch_size).cuda()
        for i in labels:
            labels_[i]=labels[i]
        labels=labels_
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # distmat.addmm_(1, -2, x, self.centers.t())
        # https://github.com/KaiyangZhou/pytorch-center-loss/issues/16
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        loss *= self.loss_weight

        return loss


# inputs = torch.rand(4, 32)
# targets = torch.Tensor([1, 1, 0, 2])
# center_loss = CenterLoss(num_classes=4, feat_dim=32, use_gpu=False, loss_weight=1.0)
# c_loss = center_loss(inputs, targets)
# print(c_loss)
#
# triplet_loss = TripletLoss()
# t_loss = triplet_loss(inputs, targets)
# print(t_loss)
