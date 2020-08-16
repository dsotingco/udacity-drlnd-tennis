"""run_tennis_ppo_agent.py
Run the tennis agent with the trained weights.
"""
from unityagents import UnityEnvironment
import numpy as np
import torch
import tennis_ppo_utils
import TennisActorCritic
import matplotlib.pyplot as plt
from collections import deque

# Environment setup
n_episodes = 2
env = UnityEnvironment(file_name="Tennis.exe")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
action_size = brain.vector_action_space_size

# Agent setup
agent = TennisActorCritic.TennisActorCritic()
agent.load_state_dict(torch.load('tennis_ppo_weights_solved.pth'))
scores_window = deque(maxlen=100)

# Run episodes
for i_episode in range(0, n_episodes):
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations.astype(np.float32)
    num_agents = len(env_info.agents)
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

         # Stop if the episode has finished 
        if np.any(dones):
            break

        # Set up for next step
        states = next_states

    # Score processing
    assert(scores.shape == (2,))
    max_agent_score = np.max(scores)
    scores_window.append(max_agent_score)
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

env.close()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores_window)), scores_window)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


