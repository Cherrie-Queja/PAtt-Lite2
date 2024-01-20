import torch
import torch.nn as nn


class WeightedAsymmetricLoss(nn.Module):
    def __init__(self, eps=1e-8, disable_torch_grad=True, weight=None):
        super(WeightedAsymmetricLoss, self).__init__()
        self.disable_torch_grad = disable_torch_grad
        self.eps = eps
        self.weight = weight

    def forward(self, x, y):
        x = torch.nn.Softmax(x)  # panr
        xs_pos = x
        xs_neg = 1 - x

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.disable_torch_grad:
            torch.set_grad_enabled(False)
        neg_weight = 1 - xs_neg
        if self.disable_torch_grad:
            torch.set_grad_enabled(True)
        loss = los_pos + neg_weight * los_neg

        if self.weight is not None:
            loss = loss * self.weight.view(1, -1)

        loss = loss.mean(dim=-1)
        return -loss.mean()

# c = WeightedAsymmetricLoss()
# x = torch.tensor([0.1, 0.1, 0.8, 0.9])
# y = torch.tensor([0, 0, 1, 1])
# loss = c(x, y)
# print(loss)
