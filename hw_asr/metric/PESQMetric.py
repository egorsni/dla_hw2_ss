from typing import List

import torch
from torch import Tensor

from hw_asr.base.base_metric import BaseMetric

from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

class PESQMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb').to('cuda')

    def __call__(self, pred, target, **kwargs):
        try:
            res = self.wb_pesq(pred, target).item()
        except Exception as e:
            return 0
        
        return self.wb_pesq(pred, target).item()
