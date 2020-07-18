""" ReplayBuffer.py """

import numpy as np
import torch
from collections import deque

class ReplayBuffer:

    def __init__(self, buffer_size=8096, batch_size=128):
        self.prob_memory = deque(maxlen=buffer_size)
        self.state_memory = deque(maxlen=buffer_size)
        self.action_memory = deque(maxlen=buffer_size)
        self.reward_memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def has_enough_samples(self):
        return ( len(self.state_memory) >= self.batch_size )

    def add_episode(self, prob_list, state_list, action_list, processed_reward_list):
        self.prob_memory.extend(prob_list)
        self.state_memory.extend(state_list)
        self.action_memory.extend(action_list)
        self.reward_memory.extend(processed_reward_list)

    def sample(self):
        assert self.has_enough_samples()
        num_samples = len(self.state_memory)
        sample_indices = np.arange(num_samples)
        np.random.shuffle(sample_indices)

        old_prob_tensor = torch.stack(list(self.prob_memory)).detach()
        old_prob_tensor_summed = torch.sum(old_prob_tensor, axis=2)
        state_tensor = torch.tensor(list(self.state_memory), dtype=torch.float).detach()
        action_tensor = torch.stack(list(self.action_memory)).detach()
        reward_tensor = torch.tensor(list(self.reward_memory), dtype=torch.float)

        old_prob_batch = old_prob_tensor_summed[0 : self.batch_size]
        state_batch = state_tensor[0 : self.batch_size]
        action_batch = action_tensor[0 : self.batch_size]
        reward_batch = reward_tensor[0 : self.batch_size]

        assert(old_prob_batch.shape == torch.Size([self.batch_size,2]))
        assert(state_batch.shape == torch.Size([self.batch_size,2,24]))
        assert(action_batch.shape == torch.Size([self.batch_size,2,2]))
        assert(reward_batch.shape == torch.Size([self.batch_size,2]))

        return (old_prob_batch, state_batch, action_batch, reward_batch)