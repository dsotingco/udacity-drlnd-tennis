"""train_tennis_ppo_agent.py
Train the Tennis agent with Proximal Policy Optimization (PPO).
"""

from unityagents import UnityEnvironment
import numpy as np
import torch
import torch.optim as optim
import tennis_ppo_utils
import TennisPpoPolicy
import ReplayBuffer
import matplotlib.pyplot as plt
from collections import deque

# Hyperparameters
learning_rate = 1e-4
num_episodes = 100
discount = 0.999
epsilon = 0.15
beta = 0.02
batch_size = 128
num_epochs_per_episode = 10
score_save_threshold = 0.25
score_solved_threshold = 0.60

# Environment setup
env = UnityEnvironment(file_name="Tennis.exe")
num_agents = 2
policy = TennisPpoPolicy.TennisPpoPolicy()
replayBuffer = ReplayBuffer.ReplayBuffer(batch_size=batch_size)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# Initialize scores, etc.
episode_scores = []
scores_window = deque(maxlen=100)

# Run episodes and train agent.
for episode in range(num_episodes):
    # Collect trajectories
    (prob_list, state_list, action_list, reward_list, episode_score) = tennis_ppo_utils.collect_trajectories(env, policy)
    processed_rewards = tennis_ppo_utils.process_rewards(reward_list)
    if(episode_score > 0.0):
        replayBuffer.add_episode(prob_list, state_list, action_list, processed_rewards.tolist())

    # Process scores
    episode_scores.append(episode_score)
    scores_window.append(episode_score)
    if episode % 10 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
            if np.mean(scores_window) >= score_save_threshold:
                torch.save(policy.state_dict(), 'tennis_ppo_weights.pth')
    if np.mean(scores_window) >= score_solved_threshold:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode-100, np.mean(scores_window)))
        break

    if replayBuffer.has_enough_samples():
        # Run training epochs
        for epoch in range(num_epochs_per_episode):
            tennis_ppo_utils.run_training_epoch(policy, optimizer, replayBuffer,
                                            discount=discount, epsilon=epsilon, beta=beta, batch_size=batch_size)

        # the clipping parameter reduces as time goes on
        epsilon*=.999
        
        # the regulation term also reduces
        # this reduces exploration in later runs
        beta*=.999

env.close()

# Plot scores
fig = plt.figure()
plt.plot(np.arange(len(episode_scores)), episode_scores)
plt.show()
