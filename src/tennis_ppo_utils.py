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

def run_episode(env, agent, traj_buffer):
    """ Run an episode of the Tennis environment with the agent. """

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

    # Initialize variables
    actions = np.zeros((num_agents, action_size))
    state_values = np.zeros((num_agents,))
    log_probs = np.zeros((num_agents,))

    # run the agents in the environment
    while True:
        # Agent 0
        (actions_0, state_value_0, log_probs_0) = agent.step(torch.tensor(states[0], dtype=torch.float))
        actions[0] = actions_0
        state_values[0] = state_value_0
        log_probs[0] = log_probs_0

        # Agent 1
        (actions_1, state_value_1, log_probs_1) = agent.step(torch.tensor(states[1], dtype=torch.float))
        actions[1] = actions_1
        state_values[1] = state_value_1
        log_probs[1] = log_probs_1

        # Sanity checks
        assert(np.isnan(actions).any() == False)
        assert(np.isnan(state_values).any() == False)
        assert(np.isnan(log_probs).any() == False)

        # Step the environment
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations.astype(np.float32)
        rewards = np.array(env_info.rewards)
        dones = env_info.local_done
        scores += env_info.rewards

        # Store Agent 0 stuff to trajectory buffer
        traj_buffer.store(states[0], 
                          actions[0], 
                          rewards[0], 
                          state_values[0],
                          log_probs[0])

        # Stop if the trajectory buffer is full
        if traj_buffer.iter >= traj_buffer.buffer_size:
            traj_buffer.finish_episode(last_value=state_values[0])
            break

        # Store Agent 1 stuff to trajectory buffer
        traj_buffer.store(states[1], 
                          actions[1], 
                          rewards[1], 
                          state_values[1],
                          log_probs[1])

        # Stop if the trajectory buffer is full
        if traj_buffer.iter >= traj_buffer.buffer_size:
            traj_buffer.finish_episode(last_value=state_values[1])
            break

        # Stop if the episode has finished 
        if np.any(dones):
            traj_buffer.finish_episode(last_value=0)
            break

        # Set up for next step
        states = next_states

    assert(scores.shape == (2,))
    max_agent_score = np.max(scores)
    #print('Max agent score this episode: {}'.format(max_agent_score))

    return max_agent_score

def calculate_policy_loss(agent, data, clip_ratio=0.1):
    # data extraction
    states         = data['states']
    actions        = data['actions']
    advantages     = data['advantages']
    log_probs_old  = data['log_probs']
    # log probability calculations
    _distribution, log_probs_new = agent.actor(states, actions)
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
    return critic_loss

def ppo_update(agent, actor_optimizer, critic_optimizer, data, clip_ratio=0.3, train_actor_iters=10, train_critic_iters=10):
    # Actor training
    for _actor_train_index in range(train_actor_iters):
        actor_optimizer.zero_grad()
        actor_loss = calculate_policy_loss(agent, data, clip_ratio=clip_ratio)
        actor_loss.backward()
        actor_optimizer.step()
    # Critic training
    for _critic_train_index in range(train_critic_iters):
        critic_optimizer.zero_grad()
        critic_loss = calculate_critic_loss(agent, data)
        critic_loss.backward()
        critic_optimizer.step()



