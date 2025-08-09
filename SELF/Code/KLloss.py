import torch.nn.functional as F
import torch.nn as nn


class KLDivLoss(nn.Module):
    def __init__(self, T):
        super(KLDivLoss, self).__init__()
        self.T = T

    def forward(self, p, q):
        q = q / (q.sum(dim=-1, keepdim=True) + 1e-30)
        q[q == 0] = 1e-30
        p[p == 0] = 1e-30

        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        loss = self.T * F.kl_div(q.log(), p, reduction='batchmean')
        return loss


