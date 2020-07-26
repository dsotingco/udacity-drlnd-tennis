import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TennisActorCritic(nn.Module):
    """ Policy model. """

    def __init__(self, state_size=24, hidden1_size=64, hidden2_size=64, action_size=2, 
                 init_std_deviation=1.0):
        super(TennisActorCritic, self).__init__()

        # Shared layers to understand the environment
        self.shared_fc1 = nn.Linear(state_size, hidden1_size)
        self.shared_fc2 = nn.Linear(hidden1_size, hidden2_size)

        # Actor layer(s), parameters
        self.actor_fc1 = nn.Linear(hidden2_size, action_size)
        # Output of Actor neural network: [mu1; mu2]
        self.std_deviations = nn.Parameter(init_std_deviation * torch.ones(1, action_size))

        # Critic layers
        self.critic_fc1 = nn.Linear(hidden2_size, 1)
        # Output of Critic neural network: scalar estimate of state value 

    def forward(self, state, actions=None, training_mode=True):
        """ Run the neural network and sample the distribution for actions. """
        assert(torch.isnan(state).any() == False)
        if training_mode:
            self.train()
        else:
            self.eval()

        # Shared layers
        x = F.leaky_relu(self.shared_fc1(state))
        x = F.leaky_relu(self.shared_fc2(x))

        # ACTOR
        means = torch.tanh(self.actor_fc1(x))
        m = torch.distributions.normal.Normal(means, self.std_deviations)

        if actions is None:
            if training_mode:
                raw_nn_actions = m.sample()
            else:
                raw_nn_actions = means
            actions = raw_nn_actions
        assert(torch.isnan(actions).any() == False)

        # NOTE: These are technically not log probabilities, but rather
        # logs of the probability density functions.
        log_probs = m.log_prob(actions)

        # CRITIC
        state_value = self.critic_fc1(x)

        return (actions, log_probs, state_value)