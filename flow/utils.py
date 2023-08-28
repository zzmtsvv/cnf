import torch
from torch import nn
from typing import Optional, Union, Sequence


class NormalizationLayer(nn.Module):
    def __init__(self,
                 state_embedding_dim: int,
                 mean: Optional[torch.Tensor] = None,
                 std: Optional[torch.Tensor] = None) -> None:
        super().__init__()

        if mean is None:
            mean = torch.zeros(state_embedding_dim, dtype=torch.float32)
        self.mean = nn.Parameter(mean, requires_grad=False)

        if std is None:
            std = torch.ones(state_embedding_dim, dtype=torch.float32)
        self.std = nn.Parameter(std, requires_grad=False)
    
    def forward(self,
                state_embedding: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if state_embedding is not None:
            state_embedding = state_embedding - self.mean
            state_embedding = state_embedding / self.std
        return state_embedding


class Concat(nn.Module):
    def __init__(self,
                 dim: Union[int, Sequence[int]]) -> None:
        super().__init__()

        self.dim = dim
    
    def forward(self, *x) -> torch.Tensor:
        return torch.cat([xx for xx in x if xx is not None], dim=self.dim)
