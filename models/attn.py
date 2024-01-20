import torch
from torch import nn


class SelfAttn(nn.Module):
    def __init__(self, n_classes=7, size_emb=768, nhead=1):
        super().__init__()
        self.n_classes = n_classes
        size_out = n_classes
        self.Q = nn.Linear(size_emb, size_out)
        self.K = nn.Linear(size_emb, size_out)
        self.V = nn.Linear(size_emb, size_out)
        self.attn_fn = nn.MultiheadAttention(size_out, num_heads=nhead, batch_first=True)

    def forward(self, x):
        q = self.Q(x).unsqueeze(1)
        k = self.K(x).unsqueeze(1)
        v = self.V(x).unsqueeze(1)
        attn_scores, _ = self.attn_fn(q, k, v)
        attn_scores = torch.softmax(attn_scores, -1)
        return attn_scores.view(-1, self.n_classes)


# self_attn = SelfAttn()
# output = self_attn(torch.Tensor(4, 768))
# print(output.shape)
