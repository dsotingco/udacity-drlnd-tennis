import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

""" Redo of my Actor/Critic PPO setup.  Now modeled after the OpenAI SpinningUp
reference implementation of PPO. """

class TennisActor(nn.Module):
    def __init__(self, state_size=24, hidden1_size=64, hidden2_size=64, action_size=2):
        super().__init__()

        # Standard deviation parameters
        log_std = -0.5 * np.ones(action_size, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        # Network layers
        self.fc1 = nn.Linear(state_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, action_size)

    def _distribution(self, states):
        x = F.leaky_relu(self.fc1(states))
        x = F.leaky_relu(self.fc2(x))
        means = torch.tanh(self.fc3(x))
        std = torch.exp(self.log_std)
        return Normal(means, std)

    def _log_prob_from_distribution(self, distribution, actions):
        return distribution.log_prob(actions).sum(axis=-1)

    def forward(self, states, actions=None):
        distribution = self._distribution(states)
        log_probs = None
        if actions is not None:
            log_probs = self._log_prob_from_distribution(distribution, actions)
        return distribution, log_probs

class TennisCritic(nn.Module):
    def __init__(self, state_size=24, hidden1_size=64, hidden2_size=64):
        super().__init__()

        # Network layers
        self.fc1 = nn.Linear(state_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, 1)

    def forward(self, states):
        x = F.leaky_relu(self.fc1(states))
        x = F.leaky_relu(self.fc2(x))
        state_values = nn.Identity(self.fc3(x))
        
        return torch.squeeze(state_values, -1)

class TennisActorCritic(nn.Module):
    def __init__(self, state_size=24, hidden1_size = 64, hidden2_size=64, action_size=2):
        super().__init()
        self.actor = TennisActor(state_size, hidden1_size, hidden2_size, action_size)
        self.critic = TennisCritic(state_size, hidden1_size, hidden2_size)

    def step(self, states):
        with torch.no_grad():
            distribution = self.actor._distribution(states)
            actions = distribution.sample()
            log_probs = self.actor._log_prob_from_distribution(distribution, actions)
            state_values = self.critic(states)
        return actions.numpy(), state_values.numpy(), log_probs.numpy()

    def act(self, states):
        return self.step(states)[0]

