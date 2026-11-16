"""
Mixed Offline-Online Replay Buffer with Symmetric Sampling.
Maintains two separate storage pools:
1. Historical demonstration buffer (B_hist)
2. Online reactor interaction buffer (B_on)
Samples symmetric minibatches (50% historical, 50% online) to enable high
update-to-data (UTD) ratios without catastrophic forgetting (penjelasan.md §5).
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List


class MixedReplayBuffer:
    """
    Replay buffer with symmetric sampling:
    Every minibatch is half historical data and half online rollout data.
    """

    def __init__(
        self,
        state_dim: int = 224,
        action_dim: int = 5,
        max_size: int = 50000,
        offline_ratio: float = 0.50,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_size = max_size
        self.offline_ratio = offline_ratio

        # Buffers for online data
        self.on_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.on_actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.on_rewards = np.zeros((max_size,), dtype=np.float32)
        self.on_next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.on_dones = np.zeros((max_size,), dtype=np.float32)
        self.on_violations = np.zeros((max_size,), dtype=np.float32)
        self.on_ptr = 0
        self.on_size = 0

        # Buffers for offline data
        self.hist_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.hist_actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.hist_rewards = np.zeros((max_size,), dtype=np.float32)
        self.hist_next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.hist_dones = np.zeros((max_size,), dtype=np.float32)
        self.hist_violations = np.zeros((max_size,), dtype=np.float32)
        self.hist_size = 0

    def load_offline_data(self, dataset):
        """Populate the historical buffer pool with transitions from D_hist."""
        n = min(len(dataset), self.max_size)
        self.hist_states[:n] = dataset.states[:n].numpy()
        self.hist_actions[:n] = dataset.actions[:n].numpy()
        self.hist_rewards[:n] = dataset.rewards[:n].numpy()
        self.hist_next_states[:n] = dataset.next_states[:n].numpy()
        self.hist_dones[:n] = dataset.dones[:n].numpy()
        self.hist_violations[:n] = dataset.violations[:n].numpy()
        self.hist_size = n
        print(f"Loaded {n} transitions into offline buffer partition.")

    def add_online(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
        violation: float = 0.0,
    ):
        """Append fresh transition from physical reactor or online rollout."""
        self.on_states[self.on_ptr] = state
        self.on_actions[self.on_ptr] = action
        self.on_rewards[self.on_ptr] = reward
        self.on_next_states[self.on_ptr] = next_state
        self.on_dones[self.on_ptr] = done
        self.on_violations[self.on_ptr] = violation

        self.on_ptr = (self.on_ptr + 1) % self.max_size
        self.on_size = min(self.on_size + 1, self.max_size)

    def sample(self, batch_size: int = 256) -> Dict[str, torch.Tensor]:
        """
        Symmetric sampling:
        Constructs a mixed_batch with batch_size // 2 from offline history
        and batch_size // 2 from online interactions.
        """
        if self.on_size == 0:
            # Fallback to pure offline sampling before online exploration begins
            idxs = np.random.randint(0, self.hist_size, size=batch_size)
            return {
                "states": torch.as_tensor(self.hist_states[idxs]),
                "actions": torch.as_tensor(self.hist_actions[idxs]),
                "rewards": torch.as_tensor(self.hist_rewards[idxs]),
                "next_states": torch.as_tensor(self.hist_next_states[idxs]),
                "dones": torch.as_tensor(self.hist_dones[idxs]),
                "violations": torch.as_tensor(self.hist_violations[idxs]),
            }

        n_hist = int(batch_size * self.offline_ratio)
        n_on = batch_size - n_hist

        hist_idxs = np.random.randint(0, self.hist_size, size=n_hist)
        on_idxs = np.random.randint(0, self.on_size, size=n_on)

        mixed_states = np.concatenate([self.hist_states[hist_idxs], self.on_states[on_idxs]], axis=0)
        mixed_actions = np.concatenate([self.hist_actions[hist_idxs], self.on_actions[on_idxs]], axis=0)
        mixed_rewards = np.concatenate([self.hist_rewards[hist_idxs], self.on_rewards[on_idxs]], axis=0)
        mixed_next_states = np.concatenate([self.hist_next_states[hist_idxs], self.on_next_states[on_idxs]], axis=0)
        mixed_dones = np.concatenate([self.hist_dones[hist_idxs], self.on_dones[on_idxs]], axis=0)
        mixed_violations = np.concatenate([self.hist_violations[hist_idxs], self.on_violations[on_idxs]], axis=0)

        return {
            "states": torch.as_tensor(mixed_states),
            "actions": torch.as_tensor(mixed_actions),
            "rewards": torch.as_tensor(mixed_rewards),
            "next_states": torch.as_tensor(mixed_next_states),
            "dones": torch.as_tensor(mixed_dones),
            "violations": torch.as_tensor(mixed_violations),
        }

    def sample_with_priority(
        self,
        batch_size: int = 128,
        priority_indices: Optional[List[int]] = None,
        num_priority: int = 16,
    ) -> Dict[str, torch.Tensor]:
        """Sample a mixed batch and append rare pre-violation online transitions when present."""
        batch = self.sample(batch_size=batch_size)
        if not priority_indices or self.on_size == 0:
            return batch
        k = min(num_priority, len(priority_indices))
        pk = np.random.choice(priority_indices, size=k, replace=True) % self.on_size
        return {
            "states": torch.cat([batch["states"], torch.as_tensor(self.on_states[pk])], dim=0),
            "actions": torch.cat([batch["actions"], torch.as_tensor(self.on_actions[pk])], dim=0),
            "rewards": torch.cat([batch["rewards"], torch.as_tensor(self.on_rewards[pk])], dim=0),
            "next_states": torch.cat([batch["next_states"], torch.as_tensor(self.on_next_states[pk])], dim=0),
            "dones": torch.cat([batch["dones"], torch.as_tensor(self.on_dones[pk])], dim=0),
            "violations": torch.cat([batch["violations"], torch.as_tensor(self.on_violations[pk])], dim=0),
        }
