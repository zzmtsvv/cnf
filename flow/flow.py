from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.distributions import Distribution

try:
    from flow_layer_interface import FlowLayer
    from utils import NormalizationLayer
except ModuleNotFoundError:
    from flow.flow_layer_interface import FlowLayer
    from flow.utils import NormalizationLayer



class Flow(nn.Module):
    def __init__(self,
                 layers: List[FlowLayer]) -> None:
        super().__init__()

        self.layers = nn.ModuleList(layers)
    
    def flow_forward(self,
                     x: torch.Tensor,
                     embedding: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x
        log_prob = torch.zeros(x.shape[:-1], dtype=torch.float32, device=x.device)

        for layer in self.layers:
            z, layer_log_prob = layer.f(z, embedding)
            log_prob += layer_log_prob
        
        return z, log_prob
    
    def flow_inverse(self,
                     z: torch.Tensor,
                     embedding: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x = z
        log_prob = torch.zeros(z.shape[:-1], dtype=torch.float32, device=z.device)

        for layer in reversed(self.layers):
            x, layer_log_prob = layer.g(x, embedding)
            log_prob += layer_log_prob
        
        return x, log_prob

    def append(self, layer: FlowLayer):
        self.layers.append(layer)
    
    def extend(self, layers: List[FlowLayer]):
        self.layers.extend(layers)


class NormalizingFlow(nn.Module):
    def __init__(self,
                 latent_distribution: Distribution,
                 flow: Flow,
                 state_embedding_dim: int,
                 state_embedding_mean: Optional[torch.Tensor] = None,
                 state_embedding_std: Optional[torch.Tensor] = None) -> None:
        super().__init__()

        self.latent_distribution = latent_distribution
        self.flow = flow

        self.norm_layer = NormalizationLayer(state_embedding_dim,
                                             state_embedding_mean,
                                             state_embedding_std)
    
    def flow_forward(self,
                     x: torch.Tensor,
                     embedding: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        state_embedding = self.norm_layer(embedding)

        z, log_prob = self.flow.flow_forward(x, state_embedding)
        log_prob += self.latent_distribution.log_prob(z).sum(-1)
        return z, log_prob
    
    def flow_inverse(self,
                     z: torch.Tensor,
                     embedding: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        state_embedding = self.norm_layer(embedding)
        log_prob = self.latent_distribution.log_prob(z).sum(-1)

        x, flow_log_prob = self.flow.flow_inverse(z, state_embedding)
        log_prob += flow_log_prob
        return x, log_prob
    
    def log_prob(self,
                 x: torch.Tensor,
                 embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, log_prob = self.flow_forward(x, embedding)
        return log_prob
    
    def sample(self,
               sample_shape: Union[int, torch.Size],
               embedding: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.latent_distribution.sample(sample_shape)
        x, log_prob = self.flow_inverse(z, embedding)
        return x, log_prob
    
    def append(self, layer: FlowLayer):
        self.flow.append(layer)
    
    def extend(self, layers: List[FlowLayer]):
        self.flow.extend(layers)
