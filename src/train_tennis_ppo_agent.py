"""train_tennis_ppo_agent.py
Train the Tennis agent with Proximal Policy Optimization (PPO).
"""

from unityagents import UnityEnvironment
import numpy as np
import torch
import torch.optim as optim
import tennis_ppo_utils
import TennisActorCritic
import TrajectoryBuffer
import matplotlib.pyplot as plt
from collections import deque

# Hyperparameters
num_episodes = 101
actor_learning_rate = 3e-4
critic_learning_rate = 1e-3
discount = 0.995
epsilon = 0.10
beta = 0.02
batch_size = 128
num_epochs_per_episode = 10
score_save_threshold = 0.25
score_solved_threshold = 0.60

# Environment setup
env = UnityEnvironment(file_name="Tennis.exe")
num_agents = 2
agent = TennisActorCritic.TennisActorCritic()
actor_optimizer = optim.Adam(agent.actor.parameters(), lr=actor_learning_rate)
critic_optimizer = optim.Adam(agent.critic.parameters(), lr=critic_learning_rate)
traj_buffer = TrajectoryBuffer.TrajectoryBuffer()

# Initialize scores, etc.
high_score = 0.0
episode_scores = []
moving_average_x = []
moving_average_y = []
scores_window = deque(maxlen=100)

# Run episodes and train agent.
for episode in range(num_episodes):
    # Run an episode
    episode_score = tennis_ppo_utils.run_episode(env, agent, traj_buffer)

    # Process scores
    if(episode_score > high_score):
        high_score = episode_score
        print("new high score:", high_score)
    episode_scores.append(episode_score)
    scores_window.append(episode_score)
    if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
            moving_average_x.append(episode)
            moving_average_y.append(np.mean(scores_window))
            if np.mean(scores_window) >= score_save_threshold:
                torch.save(agent.state_dict(), 'tennis_ppo_weights.pth')
    if np.mean(scores_window) >= score_solved_threshold:
        torch.save(agent.state_dict(), 'tennis_ppo_weights_solved.pth')
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_window)))
        break

    # PPO update
    if traj_buffer.is_full():
        data = traj_buffer.get()
        tennis_ppo_utils.ppo_update(agent, actor_optimizer, critic_optimizer, data)

env.close()

# Plot scores
fig = plt.figure()
plt.plot(np.arange(len(episode_scores)), episode_scores)
plt.plot(moving_average_x, moving_average_y)
plt.show()
