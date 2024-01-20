import torch


def Partition_Loss(f, batch_size, k):
    f = f.reshape(batch_size, k, -1)
    partition_loss = sum(sum(torch.log(1 + k / torch.var(f, dim=1)))) / (batch_size * f.shape[-1])
    return partition_loss
