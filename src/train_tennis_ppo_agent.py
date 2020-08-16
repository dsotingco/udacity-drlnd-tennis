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
num_episodes = 10001
ppo_buffer_size = 1024

actor_learning_rate = 3e-4
critic_learning_rate = 1e-3

clip_ratio = 0.3
clip_ratio_min = 0.1
clip_ratio_decay = 1

entropy_coef = 0.02
entropy_coef_decay = 0.995

train_actor_iters = 10
train_critic_iters = 10

score_solved_threshold = 0.60

# Environment setup
env = UnityEnvironment(file_name="Tennis.exe")
num_agents = 2
agent = TennisActorCritic.TennisActorCritic()
actor_optimizer = optim.Adam(agent.actor.parameters(), lr=actor_learning_rate)
critic_optimizer = optim.Adam(agent.critic.parameters(), lr=critic_learning_rate)
traj_buffer = TrajectoryBuffer.TrajectoryBuffer(buffer_size=ppo_buffer_size)

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
    if np.mean(scores_window) >= score_solved_threshold:
        torch.save(agent.state_dict(), 'tennis_ppo_weights_solved.pth')
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_window)))
        break

    # PPO update
    if traj_buffer.is_full():
        data = traj_buffer.get()
        tennis_ppo_utils.ppo_update(agent, actor_optimizer, critic_optimizer, data,
                                    clip_ratio=clip_ratio,
                                    entropy_coef=entropy_coef,
                                    train_actor_iters=train_actor_iters,
                                    train_critic_iters=train_critic_iters)
        # decay clip ratio
        decayed_clip_ratio = clip_ratio * clip_ratio_decay
        if decayed_clip_ratio >= clip_ratio_min:
            clip_ratio = decayed_clip_ratio
        # decay entropy coefficient
        entropy_coef = entropy_coef * entropy_coef_decay

env.close()

# Plot scores
fig = plt.figure()
plt.plot(np.arange(len(episode_scores)), episode_scores)
plt.plot(moving_average_x, moving_average_y)
plt.show()
