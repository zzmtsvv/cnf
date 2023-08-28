from typing import Tuple, Union, Sequence, Optional
import torch
from torch import nn
from torch.nn import functional as F

try:
    from flow_layer_interface import FlowLayer
    from utils import Concat
except ModuleNotFoundError:
    from flow.flow_layer_interface import FlowLayer
    from flow.utils import Concat


eps = 1e-6


class InvertibleLinear(FlowLayer):
    def __init__(self,
                 size: Union[int, Sequence[int]],
                 identity_initialization: bool = False) -> None:
        super().__init__()

        if identity_initialization:
            w = torch.diag(torch.ones(size, dtype=torch.float32))
        else:
            w, _ = torch.linalg.qr(torch.randn(size, size))
        
        self.weight = nn.Parameter(w)
    
    def f(self, x: torch.Tensor, *_) -> Tuple[torch.Tensor, torch.Tensor]:
        z = F.linear(x, self.weight)
        w = self.weight.to("cpu")
        log_prob = torch.linalg.slogdet(w).logabsdet

        return z, log_prob.to(x.device)
    
    def g(self, z: torch.Tensor, *_) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self.weight.to("cpu")
        w_inverse = torch.linalg.inv(w).to(z.device)

        x = F.linear(z, w_inverse)
        log_prob = torch.linalg.slogdet(w).logabsdet
        return x, log_prob.to(z.device)


class InvertibleTanh(FlowLayer):
    @staticmethod
    def f(x: torch.Tensor, *_) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.tanh(x).clamp(-1 + eps, 1 - eps)
        log_prob = torch.log(1.0 - z.pow(2)).sum(-1)
        return z, log_prob

    @staticmethod
    def g(z: torch.Tensor, *_) -> Tuple[torch.Tensor, torch.Tensor]:
        z = z.clamp(-1 + eps, 1 - eps)
        x = torch.log((1 + z) / (1 - z)) / 2
        log_prob = torch.log(1.0 - z.pow(2)).sum(-1)
        return x, log_prob


class InvertibleATanh(FlowLayer):
    @staticmethod
    def f(x: torch.Tensor, *_) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.clamp(-1 + eps, 1 - eps)
        z = torch.log((1 + x) / (1 - x)) / 2
        log_prob = -torch.log(1.0 - x.pow(2)).sum(-1)
        return z, log_prob

    @staticmethod
    def g(z: torch.Tensor, *_) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tanh(z).clamp(-1 + eps, 1 - eps)
        log_prob = -torch.log(1.0 - x.pow(2)).sum(-1)
        return x, log_prob


class Identity(FlowLayer):
    @staticmethod
    def f(x: torch.Tensor, *_) -> Tuple[torch.Tensor, float]:
        return x, 0.0

    @staticmethod
    def g(z: torch.Tensor, *_) -> Tuple[torch.Tensor, float]:
        return z, 0.0


class AffineNorm(FlowLayer):
    def __init__(self,
                 input_size: int,
                 identity_initialization: bool = False) -> None:
        super().__init__()

        self.input_size = input_size
        
        initialized = torch.tensor(identity_initialization, dtype=torch.bool)
        self.register_buffer("initialized", initialized)

        self.log_scale = nn.Parameter(torch.zeros(1, input_size), requires_grad=True)
        self.translate = nn.Parameter(torch.zeros(1, input_size), requires_grad=True)
    
    def initialize(self,
                   data: torch.Tensor,
                   forward_mode: bool) -> None:
        data = data.view(-1, self.input_size)
        log_std = torch.log(data.std(dim=0, keepdim=True) + eps)
        mean = data.mean(dim=0, keepdim=True)

        if forward_mode:
            self.log_scale.data = -log_std.data
            self.translate.data = (-mean * torch.exp(-log_std)).data
        else:
            self.log_scale.data = log_std.data
            self.translate.data = mean.data
        
        self.initialized = torch.tensor(True, dtype=torch.bool)

    def f(self, x: torch.Tensor, *_) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized and self.training:
            self.initialize(x, forward_mode=True)
        z = self.log_scale.exp() * x + self.translate
        log_prob = (self.log_scale * torch.ones_like(x)).sum(-1)
        return z, log_prob
    
    def g(self, z: torch.Tensor, *_) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized and self.training:
            self.initialize(z, forward_mode=False)
        x = (z - self.translate) * torch.exp(-self.log_scale)
        log_prob = (self.log_scale * torch.ones_like(z)).sum(-1)
        return x, log_prob


class CouplingLayer(FlowLayer):
    def __init__(self,
                 parity: bool,
                 input_dim: int,
                 state_embedding_dim: int,
                 hidden_dim: int) -> None:
        super().__init__()

        self.rescale = nn.Parameter(torch.ones(input_dim // 2))
        self.parity = parity

        self.concat = Concat(dim=-1)
        self.translation_network = nn.Sequential(
            nn.Linear(input_dim // 2 + state_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def f(self,
          x: torch.Tensor,
          embedding: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = torch.chunk(x, 2, dim=-1)
        if self.parity:
            x0, x1 = x1, x0
        
        out = self.concat(x0, embedding)
        out = self.translation_network(out)
        log_scale, translate = torch.chunk(out, 2, dim=-1)
        log_scale = self.rescale * log_scale.tanh()

        z0 = x0
        z1 = log_scale.exp() * x1 + translate

        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=-1)
        log_det = log_scale.sum(-1)

        return z, log_det
    
    def g(self,
          z: torch.Tensor,
          embedding: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        z0, z1 = torch.chunk(z, 2, dim=-1)
        if self.parity:
            z0, z1 = z1, z0
        
        out = self.concat(z0, embedding)
        out = self.translation_network(out)
        log_scale, translate = torch.chunk(out, 2, dim=-1)
        log_scale = self.rescale * log_scale.tanh()

        x0 = z0
        x1 = (z1 - translate) * torch.exp(-log_scale)

        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=-1)
        log_det = log_scale.sum(-1)

        return x, log_det
