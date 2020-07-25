"""train_tennis_ppo_agent.py
Train the Tennis agent with Proximal Policy Optimization (PPO).
"""

from unityagents import UnityEnvironment
import numpy as np
import torch
import torch.optim as optim
from ddpg_agent import Agent
import matplotlib.pyplot as plt
from collections import deque

# Hyperparameters
n_episodes = 10
random_episodes = 5
max_t = 300
print_every = 100
score_save_threshold = 0.25
score_solved_threshold = 0.60

# Environment setup
env = UnityEnvironment(file_name="Tennis.exe")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

# DDPG agent setup
agent = Agent(state_size=24, action_size=2, random_seed=2)

# Score bookkeeping
episode_scores = []
moving_average_x = []
moving_average_y = []
scores_window = deque(maxlen=100)

for episode in range(n_episodes):
    env_info = env.reset(train_mode=True)[brain_name]      # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    agent.reset()
    while True:
        if episode < random_episodes:
            actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
            actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        else:
            actions = agent.act(states)
        env_info = env.step(actions)[brain_name]           # send all actions to the environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        agent.step(states[0,:], actions[0,:], rewards[0], next_states[0,:], dones[0])
        agent.step(states[1,:], actions[1,:], rewards[1], next_states[1,:], dones[1])
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    max_agent_score = np.max(scores)
    episode_scores.append(max_agent_score)
    scores_window.append(max_agent_score)
    print('Max agent score: {}'.format(max_agent_score))
    if episode % 100 == 0:
        moving_average_x.append(episode)
        moving_average_y.append(np.mean(scores_window))
    if np.mean(scores_window) >= score_solved_threshold:
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_window)))
        break

env.close()

# Plot scores
fig = plt.figure()
plt.plot(np.arange(len(episode_scores)), episode_scores)
plt.plot(moving_average_x, moving_average_y)
plt.show()
