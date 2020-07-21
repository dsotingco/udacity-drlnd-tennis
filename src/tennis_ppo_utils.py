""" tennis_ppo_utils.py """

from unityagents import UnityEnvironment
import numpy as np
import torch

def collect_trajectories(env, policy):
    """ TODO: document the outputs """
    # initialize return variables
    prob_list = []
    state_list = []
    action_list = []
    reward_list = []
    state_value_list = []

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)

    # get initial states and scores for each agent
    states = env_info.vector_observations.astype(np.float32)
    scores = np.zeros(num_agents, dtype=np.float32)

    # run the agents in the environment
    while True:
        (actions, probs, state_value) = policy(torch.tensor(states, dtype=torch.float))
        assert(torch.isnan(actions).any() == False)
        assert(torch.isnan(probs).any() == False)
        env_info = env.step(actions.detach().numpy())[brain_name]
        next_states = env_info.vector_observations.astype(np.float32)
        rewards = np.array(env_info.rewards)
        dones = env_info.local_done
        scores += env_info.rewards

        # Type checks
        assert isinstance(probs, torch.Tensor)
        assert isinstance(states, np.ndarray)
        assert isinstance(actions, torch.Tensor)
        assert isinstance(rewards, np.ndarray)
        assert isinstance(state_value, torch.Tensor)

        # Dimension checks
        assert(probs.shape == torch.Size([2,2]))
        assert(states.shape == (2,24))
        assert(actions.shape == torch.Size([2,2]))
        assert(rewards.shape == (2,))
        assert(state_value.shape == torch.Size([2,1]))
        
        # Append results to output lists.
        prob_list.append(probs)
        state_list.append(states)
        action_list.append(actions)
        reward_list.append(rewards)
        state_value_list.append(state_value)

        # Dimension checks
        assert(len(prob_list) == len(state_list))
        assert(len(prob_list) == len(action_list))
        assert(len(prob_list) == len(reward_list))
        assert(len(prob_list) == len(state_value_list))

        # Set up for next step
        states = next_states
        if np.any(dones):
            break

    assert(scores.shape == (2,))
    max_agent_score = np.max(scores)
    # print('Max agent score this episode: {}'.format(max_agent_score))

    return prob_list, state_list, action_list, reward_list, state_value_list, max_agent_score

def calculate_discounted_future_rewards(reward_list, discount=0.995):
    # calculate discounted rewards
    discount_array = discount ** np.arange(len(reward_list))
    discounted_rewards = np.asarray(reward_list) * discount_array[:,np.newaxis]

    # calculate future discounted rewards
    discounted_future_rewards = discounted_rewards[::-1].cumsum(axis=0)[::-1]
    return discounted_future_rewards

def calculate_normalized_advantage(discounted_future_rewards, state_value_batch, discount=0.995):
    """ Calculate advantage for one run of collect_trajectories().  
    Outputs normalized, discounted, future rewards as a matrix of 
    num_timesteps rows, and num_agents columns."""
    assert(discounted_future_rewards.shape == state_value_batch.shape)
    raw_advantage = (discounted_future_rewards - state_value_batch).numpy()

    # normalize the advantage
    mean = np.mean(raw_advantage, axis=1)
    std = np.std(raw_advantage, axis=1) + 1.0e-10
    normalized_advantage = (raw_advantage - mean[:,np.newaxis]) / std[:,np.newaxis]
    assert(np.isnan(normalized_advantage).any() == False)
    return torch.tensor(normalized_advantage, dtype=torch.float)

def calculate_new_log_probs(policy, state_batch, action_batch):
    """ Calculate new log probabilities of the actions, 
        given the states.  To be used during training as the
        policy is changed by the optimizer. 
        Inputs are state and action batches as PyTorch tensors."""
    (_actions, new_prob_batch, _state_value) = policy(state_batch, actions=action_batch)
    return new_prob_batch

def calculate_probability_ratio(old_prob_batch, new_prob_batch):
    """ Calculate the PPO probability ratio. The inputs old_prob_batch
    and new_prob_batch are expected to be N x 4 PyTorch tensors, with N
    being the number of samples in the batch."""
    assert(old_prob_batch.shape == new_prob_batch.shape)
    # Note: The 4 probabilities (for 4 actions) are collapsed into a scalar to 
    # multiply by the scalar rewards.  Done in run_training_epoch() by just summing the probabilities.
    # Note that they weren't really probabilities to begin with, but rather the
    # log of the normal distributions' PDF values.
    prob_ratio = torch.exp(new_prob_batch - old_prob_batch)
    assert(torch.isnan(prob_ratio).any() == False)
    return prob_ratio

def clipped_surrogate(old_prob_batch, new_prob_batch, reward_batch,
                      discount=0.995,
                      epsilon=0.1,
                      beta=0.01):
    """ Calculate the PPO clipped surrogate function.  Inputs should be batches of
    training data, as PyTorch tensors. """
    prob_ratio = calculate_probability_ratio(old_prob_batch, new_prob_batch)
    clipped_prob_ratio = torch.clamp(prob_ratio, 1-epsilon, 1+epsilon)
    raw_loss = prob_ratio * reward_batch
    clipped_loss = clipped_prob_ratio * reward_batch
    ppo_loss = torch.min(raw_loss, clipped_loss)
    assert(torch.isnan(ppo_loss).any() == False)
    return ppo_loss

def calculate_critic_loss(discounted_future_rewards, state_value_batch, critic_discount=0.5):
    """ Calculate the loss function for the Critic. """
    # state_value_array = torch.transpose(torch.cat(state_value_list, axis=1), 0, 1).detach().numpy()
    assert(discounted_future_rewards.shape == state_value_batch.shape)
    critic_error = discounted_future_rewards - state_value_batch    # critic_error is B x 2 tensor

    # calculate the sum-squared-error, scaled by critic_discount
    critic_loss = critic_discount * torch.sum( critic_error.pow(2), axis=0 )
    assert(critic_loss.shape == torch.Size([2]))
    return critic_loss

def calculate_entropy(old_prob_batch, new_prob_batch):
    entropy = -(torch.exp(new_prob_batch) * (old_prob_batch + 1e-10) + \
              (1.0 - torch.exp(new_prob_batch)) * (1.0 - old_prob_batch + 1e-10))
    assert(torch.isnan(entropy).any() == False)
    return entropy

def run_training_epoch(policy, optimizer, replayBuffer,
                       discount=0.995,
                       epsilon=0.1,
                       beta=0.01,
                       batch_size=64):
    """ Run 1 training epoch.  Runs batches through training. """
    num_samples = len(replayBuffer.state_memory)
    num_batches = int(np.ceil(num_samples/batch_size))

    for _batch_index in range(num_batches):
        (old_prob_batch, state_batch, action_batch, discounted_future_rewards_batch, state_value_batch) = replayBuffer.sample()
        new_prob_batch_raw = calculate_new_log_probs(policy, state_batch, action_batch)
        new_prob_batch_shape = new_prob_batch_raw.shape    # should be B x 2 x 2
        assert(new_prob_batch_shape[1] == 2)
        assert(new_prob_batch_shape[2] == 2)
        new_prob_batch = torch.sum(new_prob_batch_raw, axis=2)
    
        advantage_batch = calculate_normalized_advantage(discounted_future_rewards_batch, state_value_batch)
        ppo_loss = clipped_surrogate(old_prob_batch, new_prob_batch, advantage_batch,
                                     discount=discount, epsilon=epsilon, beta=beta)
        critic_loss = calculate_critic_loss(discounted_future_rewards_batch, state_value_batch)
        entropy = calculate_entropy(old_prob_batch, new_prob_batch)
        batch_loss = -torch.mean(ppo_loss + critic_loss + beta*entropy)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

