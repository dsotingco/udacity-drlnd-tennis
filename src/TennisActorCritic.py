import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TennisActorCritic(nn.Module):
    """ Policy model. """

    def __init__(self, state_size=24, hidden1_size=128, hidden2_size=64, hidden3_size=32, action_size=2, 
                 init_std_deviation=1.0):
        super(TennisActorCritic, self).__init__()

        # Actor layers
        self.actor_fc1 = nn.Linear(state_size, hidden1_size)
        self.actor_fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.actor_fc3 = nn.Linear(hidden2_size, hidden3_size)
        self.actor_fc4 = nn.Linear(hidden3_size, action_size)
        # Output of Actor neural network: [mu1; mu2]
        self.std_deviations = nn.Parameter(init_std_deviation * torch.ones(1, action_size))

        # Critic layers
        self.critic_fc1 = nn.Linear(state_size, hidden1_size)
        self.critic_fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.critic_fc3 = nn.Linear(hidden2_size, hidden3_size)
        self.critic_fc4 = nn.Linear(hidden3_size, 1)
        # Output of Critic neural network: scalar estimate of state value 

    def forward(self, state, actions=None, training_mode=True):
        """ Run the neural network and sample the distribution for actions. """
        assert(torch.isnan(state).any() == False)
        if training_mode:
            self.train()
        else:
            self.eval()

        # ACTOR
        x = F.relu(self.actor_fc1(state))
        x = F.relu(self.actor_fc2(x))
        x = F.relu(self.actor_fc3(x))
        means = torch.tanh(self.actor_fc4(x))
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
        y = F.relu(self.critic_fc1(state))
        y = F.relu(self.critic_fc2(y))
        y = F.relu(self.critic_fc3(y))
        state_value = self.critic_fc4(y)

        return (actions, log_probs, state_value)