from typing import Tuple
import torch
from torch import nn
from abc import ABC, abstractmethod


class FlowLayer(nn.Module, ABC):
    @abstractmethod
    def f(self, x: torch.Tensor, *_) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def g(self, z: torch.Tensor, *_) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

