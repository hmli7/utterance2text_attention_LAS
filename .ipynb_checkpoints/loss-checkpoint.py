import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
class Gumbel_Softmax(nn.Module):
    """Gumbel_Softmax [Gumbel-Max trick (Gumbel, 1954; Maddison et al., 2014) + Softmax (Jang, E., Gu, S., & Poole, B., 2016)]
        source: https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    """
    def __init__(self):
        super(Gumbel_Softmax, self).__init__()
    
    def sample_gumbel_distribution(self, size, eps=1e-10):
        """sample from Gumbel Distribution (0,1)"""
        uniform_samples = torch.FloatTensor(size).uniform_(0, 1)
        return -torch.log(-torch.log(uniform_samples + eps) + eps)

    def forward(self, logits, temperature=1.0, eps=1e-10):
        noise = self.sample_gumbel_distribution(logits.size(), eps)
        if logits.is_cuda:
            noise = noise.cuda()
        y = logits + noise
        return F.softmax(y/temperature, dim=-1)
