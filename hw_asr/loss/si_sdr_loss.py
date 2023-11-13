import torch
from torch import Tensor
from torch.nn import CTCLoss
import torch.nn as nn
import numpy as np


import torch
from torch import Tensor

def si_sdr_(est, target):
    alpha = (target * est).sum() / torch.norm(target)**2
    return -20 * torch.log10(torch.norm(alpha * target) / (torch.norm(alpha * target - est) + 1e-6) + 1e-6)


class si_sdr_loss(nn.Module):
    def __init__(self):
        super(si_sdr_loss, self).__init__()
    def forward(self, pred, target) -> Tensor:
        return si_sdr_(pred, target)


# class CTCLossWrapper(CTCLoss):
#     def forward(self, log_probs, log_probs_length, text_encoded, text_encoded_length,
#                 **batch) -> Tensor:
#         log_probs_t = torch.transpose(log_probs, 0, 1)

#         return super().forward(
#             log_probs=log_probs_t,
#             targets=text_encoded,
#             input_lengths=log_probs_length,
#             target_lengths=text_encoded_length,
#         )
