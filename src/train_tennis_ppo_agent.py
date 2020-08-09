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
learning_rate = 2e-4
num_episodes = 101
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
trajectory_buffer = TrajectoryBuffer.TrajectoryBuffer(batch_size=batch_size, num_batches=num_epochs_per_episode)
optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

# Initialize scores, etc.
high_score = 0.0
episode_scores = []
moving_average_x = []
moving_average_y = []
scores_window = deque(maxlen=100)

# Run episodes and train agent.
for episode in range(num_episodes):
    # Collect trajectories
    (prob_list, state_list, action_list, reward_list, state_value_list, episode_score) = tennis_ppo_utils.collect_trajectories(env, agent)
    discounted_future_rewards = tennis_ppo_utils.calculate_discounted_future_rewards(reward_list, discount)
    trajectory_buffer.add_episode(prob_list, state_list, action_list, discounted_future_rewards.tolist(), state_value_list)
    if(episode_score > high_score):
        high_score = episode_score
        print("new high score:", high_score)

    # Process scores
    episode_scores.append(episode_score)
    scores_window.append(episode_score)
    if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
            moving_average_x.append(episode)
            moving_average_y.append(np.mean(scores_window))
            if np.mean(scores_window) >= score_save_threshold:
                torch.save(agent.state_dict(), 'tennis_ppo_weights.pth')
    if np.mean(scores_window) >= score_solved_threshold:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_window)))
        break

    if trajectory_buffer.has_enough_samples():
        # print("training...")
        # Run training epochs
        for epoch in range(num_epochs_per_episode):
            tennis_ppo_utils.run_training_epoch(agent, optimizer, trajectory_buffer,
                                            discount=discount, epsilon=epsilon, beta=beta, batch_size=batch_size)
            trajectory_buffer.clear()

        # the clipping parameter reduces as time goes on
        epsilon*=.999
        
        # the regulation term also reduces
        # this reduces exploration in later runs
        beta*=.999

env.close()

# Plot scores
fig = plt.figure()
plt.plot(np.arange(len(episode_scores)), episode_scores)
plt.plot(moving_average_x, moving_average_y)
plt.show()
