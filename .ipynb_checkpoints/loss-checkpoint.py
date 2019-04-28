import torch
import torch.nn as nn
def soft_cross_entropy(predictions, soft_targets, dim=-1, reduction=None):
    """cross entropy for soft targets"""
    logsoftmax = nn.LogSoftmax()
    if reduction == 'none':
        return torch.sum(- soft_targets * logsoftmax(predictions), dim)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(predictions), dim))

class SoftCrossEntropy(nn.Module):
    """cross entropy for soft targets"""
    def __init__(self, reduction=None):
        super(SoftCrossEntropy, self).__init__()
        self.reduction = reduction
        
    def forward(self, predictions, soft_targets, dim=-1):
        logsoftmax = nn.LogSoftmax(dim)
        if self.reduction == 'none':
            return torch.sum(- soft_targets * logsoftmax(predictions), dim)
        return torch.mean(torch.sum(- soft_targets * logsoftmax(predictions), dim))
    