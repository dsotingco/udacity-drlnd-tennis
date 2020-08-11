""" tennis_ppo_utils.py """

from unityagents import UnityEnvironment
import numpy as np
import torch
import TennisActorCritic
import TrajectoryBuffer

def normalize_advantage(advantage_batch):
    assert isinstance(advantage_batch, np.ndarray)
    mean = advantage_batch.mean()
    std  = advantage_batch.std()
    assert np.isscalar(mean)
    assert np.isscalar(std)
    normalized_advantage = (advantage_batch - mean) / std
    return normalized_advantage

def collect_trajectories(env, policy):
    """ TODO: document the outputs """
    # TODO: have 2 agents or run all inputs through 1 network?
    # initialize buffer
    trajBuffer = TrajectoryBuffer.TrajectoryBuffer()

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

def calculate_policy_loss(agent, data, clip_ratio=0.1):
    # data extraction
    states         = data['states']
    actions        = data['actions']
    advantages     = data['advantages']
    log_probs_old  = data['log_probs']
    # log probability calculations
    distribution, log_probs_new = agent.actor(states, actions)
    prob_ratio = torch.exp(log_probs_new - log_probs_old)
    clipped_prob_ratio = torch.clamp(prob_ratio, 1-clip_ratio, 1+clip_ratio)
    # loss calculations
    raw_loss = prob_ratio * advantages
    clipped_loss = clipped_prob_ratio * advantages
    ppo_loss = -( torch.min(raw_loss, clipped_loss) ).mean()
    return ppo_loss

def calculate_critic_loss(agent, data):
    states = data['states']
    returns = data['returns']
    state_values = agent.critic(states)
    critic_loss = ( (state_values - returns)**2 ).mean()


