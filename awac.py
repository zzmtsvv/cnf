from typing import Dict, Union
from copy import deepcopy
import torch
from torch.nn import functional as F
from config import cnf_config
from modules import Actor, EnsembledCritic
from flow.flow import NormalizingFlow


_Number = Union[float, int]


class AWAC:
    def __init__(self,
                 cfg: cnf_config,
                 actor: Actor,
                 critic: EnsembledCritic,
                 flow: NormalizingFlow) -> None:
        self.cfg = cfg
        self.device = cfg.device

        self.actor = actor.to(self.device)
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)

        self.critic = critic.to(self.device)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=cfg.critic_lr)

        with torch.no_grad():
            self.critic_target = deepcopy(critic).to(self.device)
        
        self.flow = flow.to(self.device)
        
        self.tau = cfg.tau
        self.gamma = cfg.gamma
        self.awac_lambda = cfg.awac_lambda
        self.exp_adv_max = cfg.exp_adv_max
        
        self.total_iterations = 0
    
    def train(self,
              states: torch.Tensor,
              actions: torch.Tensor,
              rewards: torch.Tensor,
              next_states: torch.Tensor,
              dones: torch.Tensor) -> Dict[str, _Number]:
        self.total_iterations += 1

        critic_loss = self.critic_loss(states, actions, rewards, next_states, dones)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = self.actor_loss(states, actions)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.soft_critic_update()

        return {
            "awac/critic_loss": critic_loss.item(),
            "awac/actor_loss": actor_loss.item()
        }
        
    
    def actor_loss(self,
                   states: torch.Tensor,
                   actions: torch.Tensor) -> torch.Tensor:
        pi_action, pi_log_prob = self.actor(states)
        pi_action, _ = self.flow.flow_inverse(pi_action, states)

        with torch.no_grad():

            values = self.critic(states, pi_action.detach()).min(0).values
            q_values = self.critic(states, actions).min(0).values

            advantage = q_values - values
            weights = torch.clamp_max(torch.exp(advantage / self.awac_lambda), self.exp_adv_max)
        
        print(pi_action.shape, actions.shape, weights.shape)

        action_diff = F.l1_loss(pi_action, actions, reduction="none")
        loss = (weights.unsqueeze(1) * action_diff).mean()
        return loss

    def critic_loss(self,
                    states: torch.Tensor,
                    actions: torch.Tensor,
                    rewards: torch.Tensor,
                    next_states: torch.Tensor,
                    dones: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_actions, _ = self.actor(next_states)

            next_actions, _ = self.flow.flow_inverse(next_actions, next_states)
            q_next = self.critic_target(next_states, next_actions).min(0).values
            q_target = rewards + self.gamma * (1.0 - dones) * q_next

            tgt_q = rewards + self.gamma * (1 - dones) * q_next.unsqueeze(-1)
            tgt_q.squeeze_(1)
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, tgt_q)
        
        return critic_loss

    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * tgt_param.data)

