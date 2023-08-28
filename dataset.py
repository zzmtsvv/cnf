from typing import Dict, Tuple
import os
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def from_json(json_file: str) -> Dict[str, np.ndarray]:
    if not json_file.endswith('.json'):
        json_file = json_file + '.json'

    json_file = os.path.join("json_datasets", json_file)
    output = dict()
    
    with open(json_file) as f:
        dataset = json.load(f)
    
    for k, v in dataset.items():
        v = np.array(v)
        if k != "terminals":
            v = v.astype(np.float32)
    
        output[k] = v
        
    return output


class TensorDataset(Dataset[Tuple[torch.Tensor, ...]]):
    tensors: Tuple[torch.Tensor, ...]

    def __init__(self, *tensors: torch.Tensor) -> None:
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


def make_dataloader(batch_size: int,
                    env_name: str,
                    for_behavior_cloning: bool = False) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    eps = 1e-5
    actions_low = -1 + 2 * eps
    actions_high = 1 - 2 * eps
    test_size = 0.1

    dataset = from_json(env_name)

    if for_behavior_cloning:
        states, actions = [
            torch.tensor(dataset[key], dtype=torch.float32)
            for key in ("observations", "actions")
        ]
    else:
        states, actions, rewards, dones, next_states = [
            torch.tensor(dataset[key], dtype=torch.float32)
            for key in ['observations', 'actions', 'rewards', 'terminals', 'next_observations']
        ]
        rewards = rewards[..., None]
        dones = dones[..., None]
    
    actions = actions.clamp(actions_low, actions_high)

    states_mean = states.mean(dim=0)
    states_std = states.std(dim=0)

    if for_behavior_cloning:
        tensors = [actions, states]
    else:
        tensors = [states, actions, rewards, next_states, dones]
    
    data = train_test_split(*tensors, test_size=test_size)
    train_data, test_data = data[::2], data[1::2]

    train_dataset = TensorDataset(*train_data)
    test_dataset = TensorDataset(*test_data)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True)
    return train_loader, test_loader, states_mean, states_std
