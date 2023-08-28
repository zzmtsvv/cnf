from tqdm import tqdm
import torch
from torch import nn
import random
import numpy as np
from torch.distributions import Uniform, Normal
from typing import List, Optional, Dict

import wandb

from config import cnf_config
from dataset import make_dataloader
from replay_buffer import ReplayBuffer
from awac import AWAC
from modules import Actor, EnsembledCritic

from flow.flow import NormalizingFlow, Flow
from flow.flow_layer_interface import FlowLayer
from flow.invertible import AffineNorm, InvertibleLinear, CouplingLayer, InvertibleATanh, InvertibleTanh, Identity


class Logger:
    def log(self, something, step):
        pass


class CNFTrainer:
    def __init__(self,
                 cfg=cnf_config) -> None:
        self.cfg = cfg
        self.device = cfg.device
        seed = cfg.seed

        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def fit(self):
        print(f"Training starts on {self.device}🚀")

        with wandb.init(project="meow", entity="zzmtsvv", group=f"awac_{self.cfg.group}", name=self.cfg.name):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})

            self.train_flow()
            # self.flow.load_state_dict(torch.load("flow.pt"))

            actor = Actor(17, 6, self.cfg.hidden_dim).to(self.device)
            critic = EnsembledCritic(17, 6, self.cfg.hidden_dim).to(self.device)
            buffer = ReplayBuffer(17, 6)
            buffer.from_json(self.cfg.dataset_name)

            self.awac = AWAC(self.cfg,
                             actor,
                             critic,
                             self.flow)
            
            self.train_awac(buffer)
    
    def train_awac(self,
                   buffer: ReplayBuffer):
        
        for t in tqdm(range(self.cfg.max_timesteps), desc="AWAC steps"):
            batch = buffer.sample(self.cfg.batch_size)
            
            states, actions, rewards, next_states, dones = [x.to(self.device) for x in batch]
            
            logging_dict = self.awac.train(states,
                                           actions,
                                           rewards,
                                           next_states,
                                           dones)
            
            wandb.log(logging_dict, step=self.awac.total_iterations)
    
    def train_flow(self):
        train_dataloader, valid_dataloader, state_mean, state_std = make_dataloader(self.cfg.flow_batch_size,
                                                                                    self.cfg.dataset_name,
                                                                                    for_behavior_cloning=True)
        action_dim, state_dim = 6, 17

        self.flow = self.make_flow(num_layers=self.cfg.flow_num_layers,
                                   data_dim=action_dim,
                                   state_embedding_dim=state_dim,
                                   hidden_dim=self.cfg.hidden_dim,
                                   add_atanh=self.cfg.use_atanh,
                                   uniform_latent=self.cfg.uniform_latent,
                                   state_embedding_mean=state_mean,
                                   state_embedding_std=state_std)
        return
        
        self.flow_optimizer = torch.optim.AdamW(self.flow.parameters(),
                                                lr=self.cfg.flow_lr,
                                                weight_decay=self.cfg.flow_wd)
        self.init_flow(train_dataloader)
        self.train_iterations = 0
        self.val_iterations = 0


        # with wandb.init(project=self.cfg.project, entity="zzmtsvv", group=self.cfg.group, name=self.cfg.name) as logger:
            # logger.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})

        for epoch in range(self.cfg.flow_num_epochs):
            self.train_epoch(epoch,
                             train_dataloader)
            self.valid_epoch(epoch,
                             valid_dataloader)

    def train_epoch(self,
                    epoch,
                    data_loader) -> int:
        self.flow.train()
        p_bar = tqdm(data_loader, desc=f"NF train_{epoch + 1}", ncols=80)
        for batch in p_bar:
            batch = [b.to(self.device) for b in batch]
            res = self.optimize_flow(batch)
            self.train_iterations += 1

            # res.update({"flow/train_consumed_data": consumed_data})
#             wandb.log(res, step=self.train_iterations)
    
    def valid_batch(self, batch: List[torch.Tensor]) -> Dict[str, float]:
        actions, states = batch
        batch_size = states.size(0)

        states_for_sampling = torch.cat([states for _ in range(self.cfg.flow_num_validation_samples)], dim=0)
        action_samples, samples_log_prob = self.flow.sample(states_for_sampling.size()[:-1], states_for_sampling)

        actions_for_l1 = torch.cat([actions for _ in range(self.cfg.flow_num_validation_samples)], dim=0)
        mae = torch.abs(actions_for_l1 - action_samples).mean(-1)
        mae = mae.view(self.cfg.flow_num_validation_samples, -1)

        mean_samples_std = action_samples.view(self.cfg.flow_num_validation_samples, batch_size, -1).std(0).mean()
        mean_mae = mae.mean()
        min_mae = torch.min(mae, dim=0)[0].mean()
        samples_mean_neg_log_prob = -samples_log_prob.mean()  # negative to be on the same scale as loss

        valid_mean_neg_log_prob = -self.flow.log_prob(actions, states).mean()

        return {
            'flow/mean_samples_std': mean_samples_std.item(),
            'flow/mean_mae': mean_mae.item(),
            'flow/min_mae': min_mae.item(),
            'flow/samples_mean_neg_log_prob': samples_mean_neg_log_prob.item(),
            'flow/valid_mean_neg_log_prob': valid_mean_neg_log_prob.item()
        }

    def valid_epoch(self,
                    epoch,
                    data_loader) -> int:
        self.flow.eval()

        with torch.no_grad():
            p_bar = tqdm(data_loader, desc=f"NF val_{epoch + 1}", ncols=80)
            for batch in p_bar:
                batch = [b.to(self.device) for b in batch]
                res = self.valid_batch(batch)
                self.val_iterations += 1
                
                wandb.log(res, step=self.val_iterations)

    def optimize_flow(self,
                      data: List[torch.Tensor],
                      clip_grad_norm: float = 0.1) -> Dict[str, float]:
        log_prob = self.flow.log_prob(*data)
        loss = -log_prob.mean()

        self.flow_optimizer.zero_grad()
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(self.flow.parameters(), clip_grad_norm)

        self.flow_optimizer.step()

        return {
            "flow/train_loss": loss.item(),
            "flow/grad_norm": grad_norm.item()
        }

    def init_flow(self, dataloader):
        for batch in dataloader:
            if isinstance(batch, torch.Tensor):
                batch = [batch]
            batch = [b.to(self.device) for b in batch]
            with torch.no_grad():
                self.flow.log_prob(*batch)

    def make_flow_layers(self,
                         num_layers: int,
                         data_dim: int,
                         state_embedding_dim: int,
                         hidden_dim: int,
                         identity_initialization: bool = False) -> List[FlowLayer]:
        layers = []

        for i in range(num_layers):
            layers.append(AffineNorm(data_dim, identity_initialization))
            layers.append(InvertibleLinear(data_dim, identity_initialization))
            layers.append(CouplingLayer(i % 2 == 0, data_dim, state_embedding_dim, hidden_dim))
        
        return layers
    
    def make_flow(self,
                  num_layers: int,
                  data_dim: int,
                  state_embedding_dim: int,
                  hidden_dim: int,
                  add_atanh: bool,
                  uniform_latent: bool,
                  state_embedding_mean: Optional[torch.Tensor] = None,
                  state_embedding_std: Optional[torch.Tensor] = None,
                  identity_initialization: bool = False) -> NormalizingFlow:
        layers = []
        if add_atanh:
            layers.append(InvertibleATanh())
        else:
            layers.append(Identity())
        layers = self.make_flow_layers(num_layers, data_dim, state_embedding_dim, hidden_dim, identity_initialization)
        flow = Flow(layers)

        if uniform_latent:
            flow.append(InvertibleTanh())
            
            latent_min = -torch.ones(data_dim, dtype=torch.float32, device=self.device, requires_grad=False)
            latent_max = torch.ones(data_dim, dtype=torch.float32, device=self.device, requires_grad=False)
            latent_distribution = Uniform(latent_min, latent_max)
        else:
            latent_mean = torch.zeros(data_dim, dtype=torch.float32, device=self.device, requires_grad=False)
            latent_std = torch.ones(data_dim, dtype=torch.float32, device=self.device, requires_grad=False)
            latent_distribution = Normal(latent_mean, latent_std)
        
        normalizing_flow = NormalizingFlow(latent_distribution,
                                           flow,
                                           state_embedding_dim,
                                           state_embedding_mean,
                                           state_embedding_std).to(self.device)
        return normalizing_flow

