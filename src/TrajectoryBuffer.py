""" TrajectoryBuffer.py """

import numpy as np
import torch
import tennis_ppo_utils

class TrajectoryBuffer:
    """
    A class for storing samples for training by a PPO agent.  
    Modeled after the PPOBuffer from OpenAI SpinningUp.
    """

    def __init__(self, state_size=24, action_size=2, buffer_size, 
                 discount_gamma=0.99, gae_lambda=0.95):
        self.state_memory       = np.zeros( (size, state_size), dtype=np.float32 )
        self.action_memory      = np.zeros( (size, action_size), dtype=np.float32 )
        self.advantage_memory   = np.zeros(size, dtype=np.float32)
        self.reward_memory      = np.zeros(size, dtype=np.float32)
        self.returns_memory     = np.zeros(size, dtype=np.float32)
        self.state_value_memory = np.zeros(size, dtype=np.float32)
        self.log_prob_memory    = np.zeros(size, dtype=np.float32)

        self.discount_gamma = discount_gamma
        self.gae_lambda = gae_lambda

        self.iter = 0
        self.episode_start_index = 0
        self.buffer_size = buffer_size

    def store(self, states, actions, rewards, state_values, log_probs):
        assert self.iter < self.buffer_size
        self.state_memory      [ self.iter ] = states
        self.action_memory     [ self.iter ] = actions
        self.reward_memory     [ self.iter ] = rewards
        self.state_value_memory[ self.iter ] = state_values
        self.log_prob_memory   [ self.iter ] = log_probs
        self.iter += 1

    def finish_episode(self, last_value=0):
        """
        Function to be used at the end of an episode, or when the buffer is full.
        Resets the buffer iterator and also performs calculations that can
        only be done at the end, specifically advantages and returns.

        The last_value argument should be 0 if the episode ended from the agent
        having reached a terminal state, and otherwise whould be V(s_T), the value
        function estimated for the last state.
        """
        traj_slice = slice(self.episode_start_index, self.iter)
        rewards = np.append(self.reward_memory[ traj_slice ], last_value)
        state_values = np.append(self.state_value_memory[ traj_slice ], last_value)

        # Generalized Advantage Estimation (GAE)
        deltas = rewards[:-1] + self.discount_gamma * state_values[1:] - state_values[:-1]
        self.advantage_memory[ traj_slice ] = tennis_ppo_utils.discount_cumsum(deltas, self.discount_gamma * self.gae_lambda)

        # Calculate returns (to be used in critic loss, for training the Critic)
        self.returns_memory[ traj_slice ] = tennis_ppo_utils.discount_cumsum(rewards, self.discount_gamma)[:-1]

        self.episode_start_index = self.iter

    def get(self):
        assert self.iter == self.buffer_size
        # Reset iterators
        self.iter = 0
        self.episode_start_index = 0

        # Normalize advantage and replace in advantage_memory
        normalized_advantage = tennis_ppo_utils.normalize_advantage( self.advantage_memory )
        self.advantage_memory = normalized_advantage

        # Convert to PyTorch tensors and return data in dictionary
        data_np = dict( states     = self.state_memory,
                        actions    = self.action_memory,
                        returns    = self.returns_memory,
                        advantages = self.advantage_memory,
                        log_probs  = self.log_prob_memory )
        data_torch = { key: torch.as_tensor(value, dtype=torch.float32) for key,value in data_np.items() }
        return data_torch







