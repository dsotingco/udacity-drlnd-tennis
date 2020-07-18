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
        (actions, probs) = policy(torch.tensor(states, dtype=torch.float))
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

        # Dimension checks
        assert(probs.shape == torch.Size([2,2]))
        assert(states.shape == (2,24))
        assert(actions.shape == torch.Size([2,2]))
        assert(rewards.shape == (2,24))
        
        # Append results to output lists.
        prob_list.append(probs)
        state_list.append(states)
        action_list.append(actions)
        reward_list.append(rewards)

        # Dimension checks
        assert(len(prob_list) == len(state_list))
        assert(len(prob_list) == len(action_list))
        assert(len(prob_list) == len(reward_list))

        # Set up for next step
        states = next_states
        if np.any(dones):
            break

    assert(scores.shape == (2,))
    max_agent_score = np.max(scores)
    print('Max agent score this episode: {}'.format(max_agent_score))

    return prob_list, state_list, action_list, reward_list, max_agent_score

def process_rewards(reward_list, discount=0.995):
    """ Process the rewards for one run of collect_trajectories().  
    Outputs normalized, discounted, future rewards as a matrix of 
    num_timesteps rows, and num_agents columns."""
    # calculate discounted rewards
    discount_array = discount ** np.arange(len(reward_list))
    discounted_rewards = np.asarray(reward_list) * discount_array[:,np.newaxis]

    # calculate future discounted rewards
    future_rewards = discounted_rewards[::-1].cumsum(axis=0)[::-1]

    # normalize the future discounted rewards
    mean = np.mean(future_rewards, axis=1)
    std = np.std(future_rewards, axis=1) + 1.0e-10
    normalized_rewards = (future_rewards - mean[:,np.newaxis]) / std[:,np.newaxis]
    assert(np.isnan(normalized_rewards).any() == False)
    return normalized_rewards

def calculate_new_log_probs(policy, state_batch, action_batch):
    """ Calculate new log probabilities of the actions, 
        given the states.  To be used during training as the
        policy is changed by the optimizer. 
        Inputs are state and action batches as PyTorch tensors."""
    (_actions, new_prob_batch) = policy(state_batch, actions=action_batch)
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

def calculate_entropy(old_prob_batch, new_prob_batch):
    entropy = -(torch.exp(new_prob_batch) * (old_prob_batch + 1e-10) + \
              (1.0 - torch.exp(new_prob_batch)) * (1.0 - old_prob_batch + 1e-10))
    assert(torch.isnan(entropy).any() == False)
    return entropy

def run_training_epoch(policy, optimizer, old_prob_list, state_list, action_list, reward_list,
                       discount=0.995,
                       epsilon=0.1,
                       beta=0.01,
                       batch_size=64):
    """ Run 1 training epoch.  Takes in the output lists from 1 run of collect_trajectories()
    for a single episode.  Breaks up the lists into batches and then runs the batches through 
    training. """
    num_samples = len(state_list)
    num_batches = int(np.ceil(num_samples/batch_size))
    sample_indices = np.arange(num_samples)
    np.random.shuffle(sample_indices)

    old_prob_tensor = torch.stack(old_prob_list).detach()                           # T x 2 x  2 (T x num_agents x action_size)
    old_prob_tensor_summed = torch.sum(old_prob_tensor, axis=2)                     # T x 2      (T x num_agents)
    state_tensor = torch.tensor(state_list, dtype=torch.float).detach()             # T x 2 x 24 (T x num_agents x state_size)
    action_tensor = torch.stack(action_list).detach()                               # T x 2 x  2 (T x num_agents x action_size)
    reward_tensor = torch.tensor(process_rewards(reward_list), dtype=torch.float)   # T x 2      (T x num_agents)

    # TODO: see if T is constant per episode for this environment
    # assert(old_prob_tensor.shape == torch.Size([1001,20,4]))
    # assert(old_prob_tensor_summed.shape == torch.Size([1001,20]))
    # assert(state_tensor.shape == torch.Size([1001,20,33]))
    # assert(action_tensor.shape == torch.Size([1001,20,4]))
    # assert(reward_tensor.shape == torch.Size([1001,20]))
    # assert(torch.isnan(old_prob_tensor).any() == False)
    # assert(torch.isnan(old_prob_tensor_summed).any() == False)
    # assert(torch.isnan(state_tensor).any() == False)
    # assert(torch.isnan(action_tensor).any() == False)
    # assert(torch.isnan(reward_tensor).any() == False)

    for batch_index in range(num_batches):
        sample_start_index = batch_index * batch_size
        sample_end_index = sample_start_index + batch_size
        batch_sample_indices = sample_indices[sample_start_index : sample_end_index]

        old_prob_batch = old_prob_tensor_summed[batch_sample_indices]
        state_batch = state_tensor[batch_sample_indices]
        action_batch = action_tensor[batch_sample_indices]
        reward_batch = reward_tensor[batch_sample_indices]
        new_prob_batch_raw = calculate_new_log_probs(policy, state_batch, action_batch)
        new_prob_batch_shape = new_prob_batch_raw.shape    # should be B x 20 x 4
        assert(new_prob_batch_shape[1] == 2)
        assert(new_prob_batch_shape[2] == 2)
        new_prob_batch = torch.sum(new_prob_batch_raw, axis=2)
    
        ppo_loss = clipped_surrogate(old_prob_batch, new_prob_batch, reward_batch,
                                     discount=discount, epsilon=epsilon, beta=beta)
        entropy = calculate_entropy(old_prob_batch, new_prob_batch)
        batch_loss = -torch.mean(ppo_loss + beta*entropy)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

