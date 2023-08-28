from dataclasses import dataclass
import torch


@dataclass
class cnf_config:
    # Experiment
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name: str = "halfcheetah-medium-v2"
    seed: int = 42

    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 512
    buffer_size: int = 1000000
    gamma: float = 0.99
    hidden_dim: int = 256
    max_action: float = 1.0
    max_timesteps: int = int(2e5)
    tau: float = 2e-3
    exp_adv_max: float = 100.0
    awac_lambda: float = 0.3333

    flow_num_layers: int = 12
    flow_num_epochs: int = 100
    use_atanh: bool = True
    flow_lr: float = 5e-4
    flow_wd: float = 1e-4
    flow_batch_size: int = 1024
    flow_num_validation_samples: int = 100
    uniform_latent: bool = True

    project: str = "CNF"
    group: str = dataset_name
    name: str = dataset_name + "_" + str(seed)