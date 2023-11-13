from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_cer

from torchmetrics.audio import ScaleInvariantSignalDistortionRatio


class SiSDRMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_sdr = ScaleInvariantSignalDistortionRatio().to('cuda')

    def __call__(self, pred, target, **kwargs):
        return self.si_sdr(pred, target).item()
