from typing import Optional

import prettytable
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.grl import WarmStartGradientReverseLayer

def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct


def accuracy(output, target, topk=(1,)):
    r"""
    Computes the accuracy over the k top predictions for the specified values of k

    Args:
        output (tensor): Classification outputs, :math:`(N, C)` where `C = number of classes`
        target (tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`
        topk (sequence[int]): A list of top-N number.

    Returns:
        Top-N accuracies (N :math:`\in` topK).
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, target, output):
        """
        Update confusion matrix.

        Args:
            target: ground truth
            output: predictions of models

        Shape:
            - target: :math:`(minibatch, C)` where C means the number of classes.
            - output: :math:`(minibatch, C)` where C means the number of classes.
        """
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + output[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        """compute global accuracy, per-class accuracy and per-class IoU"""
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    # def reduce_from_all_processes(self):
    #     if not torch.distributed.is_available():
    #         return
    #     if not torch.distributed.is_initialized():
    #         return
    #     torch.distributed.barrier()
    #     torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)

    def format(self, classes: list):
        """Get the accuracy and IoU for each class in the table format"""
        acc_global, acc, iu = self.compute()

        table = prettytable.PrettyTable(["class", "acc", "iou"])
        for i, class_name, per_acc, per_iu in zip(range(len(classes)), classes, (acc * 100).tolist(), (iu * 100).tolist()):
            table.add_row([class_name, per_acc, per_iu])

        return 'global correct: {:.1f}\nmean correct:{:.1f}\nmean IoU: {:.1f}\n{}'.format(
            acc_global.item() * 100, acc.mean().item() * 100, iu.mean().item() * 100, table.get_string())

class DomainAdversarialLoss(nn.Module):
    r"""
    The Domain Adversarial Loss proposed in
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

    Domain adversarial loss measures the domain discrepancy through training a domain discriminator.
    Given domain discriminator :math:`D`, feature representation :math:`f`, the definition of DANN loss is

    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) = \mathbb{E}_{x_i^s \sim \mathcal{D}_s} \text{log}[D(f_i^s)]
            + \mathbb{E}_{x_j^t \sim \mathcal{D}_t} \text{log}[1-D(f_j^t)].

    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of features. Its input shape is (N, F) and output shape is (N, 1)
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        grl (WarmStartGradientReverseLayer, optional): Default: None.

    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`
        - w_s (tensor, optional): a rescaling weight given to each instance from source domain.
        - w_t (tensor, optional): a rescaling weight given to each instance from target domain.

    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(N, )`.

    Examples::

        # >>> from tllib.modules.domain_discriminator import DomainDiscriminator
        # >>> discriminator = DomainDiscriminator(in_feature=1024, hidden_size=1024)
        # >>> loss = DomainAdversarialLoss(discriminator, reduction='mean')
        # >>> # features from source domain and target domain
        # >>> f_s, f_t = torch.randn(20, 1024), torch.randn(20, 1024)
        # >>> # If you want to assign different weights to each instance, you should pass in w_s and w_t
        # >>> w_s, w_t = torch.randn(20), torch.randn(20)
        # >>> output = loss(f_s, f_t, w_s, w_t)
    """

    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean',
                 grl: Optional = None, sigmoid=True):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True) if grl is None else grl
        self.domain_discriminator = domain_discriminator
        self.sigmoid = sigmoid
        self.reduction = reduction
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        if self.sigmoid:
            d_s, d_t = d.chunk(2, dim=0)
            d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
            d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
            self.domain_discriminator_accuracy = 0.5 * (
                        binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))

            if w_s is None:
                w_s = torch.ones_like(d_label_s)
            if w_t is None:
                w_t = torch.ones_like(d_label_t)
            return 0.5 * (
                F.binary_cross_entropy(d_s, d_label_s, weight=w_s.view_as(d_s), reduction=self.reduction) +
                F.binary_cross_entropy(d_t, d_label_t, weight=w_t.view_as(d_t), reduction=self.reduction)
            )
        else:
            d_label = torch.cat((
                torch.ones((f_s.size(0),)).to(f_s.device),
                torch.zeros((f_t.size(0),)).to(f_t.device),
            )).long()
            if w_s is None:
                w_s = torch.ones((f_s.size(0),)).to(f_s.device)
            if w_t is None:
                w_t = torch.ones((f_t.size(0),)).to(f_t.device)
            self.domain_discriminator_accuracy = accuracy(d, d_label)
            loss = F.cross_entropy(d, d_label, reduction='none') * torch.cat([w_s, w_t], dim=0)
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            elif self.reduction == "none":
                return loss
            else:
                raise NotImplementedError(self.reduction)