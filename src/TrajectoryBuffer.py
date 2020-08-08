""" TrajectoryBuffer.py """

import numpy as np
import torch
from collections import deque

class TrajectoryBuffer:

    def __init__(self, batch_size=128, num_batches=10):
        buffer_size = batch_size * num_batches
        self.prob_memory = deque(maxlen=buffer_size)
        self.state_memory = deque(maxlen=buffer_size)
        self.action_memory = deque(maxlen=buffer_size)
        self.advantage_memory = deque(maxlen=buffer_size)
        self.returns_memory = deque(maxlen=buffer_size)
        self.state_value_memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.sampled_count = 0
        self.sample_indices = np.arange(buffer_size)

    def has_enough_samples(self):
        return ( len(self.state_memory) >= ( self.batch_size * self.num_batches ) )

    def clear(self):
        self.prob_memory.clear()
        self.state_memory.clear()
        self.action_memory.clear()
        self.advantage_memory.clear()
        self.returns_memory.clear()
        self.state_value_memory.clear()
        self.sampled_count = 0

    def add_episode(self, prob_list, state_list, action_list, advantage_list, returns_list, state_value_list):
        self.prob_memory.extend(prob_list)
        self.state_memory.extend(state_list)
        self.action_memory.extend(action_list)
        self.advantage_memory.extend(advantage_list)
        self.returns_memory.extend(returns_list)
        self.state_value_memory.extend(state_value_list)

    def sample(self):
        assert self.has_enough_samples()
        if self.sampled_count == 0:
            np.random.shuffle(self.sample_indices)
        sample_start_index = self.sampled_count * self.batch_size
        sample_end_index = sample_start_index + self.batch_size
        self.sampled_count += 1

        old_prob_tensor = torch.stack(list(self.prob_memory)).detach()
        old_prob_tensor_summed = torch.sum(old_prob_tensor, dim=2)    # TODO: is dim=2 correct?
        state_tensor = torch.tensor(list(self.state_memory), dtype=torch.float).detach()
        action_tensor = torch.stack(list(self.action_memory)).detach()
        advantage_tensor = torch.stack(list(self.advantage_memory)).detach()
        #returns_tensor = torch.transpose(torch.cat(list(self.returns_memory), dim=1), 0, 1).detach()
        returns_tensor = torch.tensor(list(self.returns_memory), dtype=torch.float).detach()
        state_value_tensor = torch.stack(list(self.state_value_memory)).detach()

        old_prob_batch = old_prob_tensor_summed[sample_start_index : sample_end_index]
        state_batch = state_tensor[sample_start_index : sample_end_index]
        action_batch = action_tensor[sample_start_index : sample_end_index]
        advantage_batch = advantage_tensor[sample_start_index : sample_end_index]
        returns_batch = returns_tensor[sample_start_index : sample_end_index]
        state_value_batch = state_value_tensor[sample_start_index : sample_end_index]
        state_value_batch = torch.squeeze(state_value_batch)

        assert(old_prob_batch.shape == torch.Size([self.batch_size,2]))
        assert(state_batch.shape == torch.Size([self.batch_size,2,24]))
        assert(action_batch.shape == torch.Size([self.batch_size,2,2]))
        assert(advantage_batch.shape == torch.Size([self.batch_size,2]))
        assert(returns_batch.shape == torch.Size([self.batch_size,2]))
        assert(state_value_batch.shape == torch.Size([self.batch_size,2]))

        return (old_prob_batch, state_batch, action_batch, advantage_batch, returns_batch, state_value_batch)