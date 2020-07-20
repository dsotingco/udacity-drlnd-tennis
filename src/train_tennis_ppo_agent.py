"""train_tennis_ppo_agent.py
Train the Tennis agent with Proximal Policy Optimization (PPO).
"""

from unityagents import UnityEnvironment
import numpy as np
import torch
import torch.optim as optim
import tennis_ppo_utils
import TennisActorCritic
import ReplayBuffer
import matplotlib.pyplot as plt
from collections import deque

# Hyperparameters
learning_rate = 2e-4
num_episodes = 10
discount = 0.999
epsilon = 0.15
beta = 0.02
batch_size = 128
replay_buffer_store_threshold = -1.0
high_score_buffer_sample_limit = 100
num_epochs_per_episode = 10
score_save_threshold = 0.25
score_solved_threshold = 0.60
use_high_score_replay_buffer = False

# Environment setup
env = UnityEnvironment(file_name="Tennis.exe")
num_agents = 2
agent = TennisActorCritic.TennisActorCritic()
replay_buffer = ReplayBuffer.ReplayBuffer(batch_size=batch_size)
high_score_replay_buffer = ReplayBuffer.ReplayBuffer(batch_size=batch_size)
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
    actor_critic_advantage = tennis_ppo_utils.calculate_advantage(reward_list, state_value_list)
    if(episode_score > replay_buffer_store_threshold):
        replay_buffer.add_episode(prob_list, state_list, action_list, actor_critic_advantage.tolist())
    if(episode_score >= high_score):
        high_score = episode_score
        print("high score:", high_score)
        high_score_replay_buffer.add_episode(prob_list, state_list, action_list, actor_critic_advantage.tolist())

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

    if replay_buffer.has_enough_samples():
        # print("training...")
        # Run training epochs
        for epoch in range(num_epochs_per_episode):
            tennis_ppo_utils.run_training_epoch(agent, optimizer, replay_buffer,
                                            discount=discount, epsilon=epsilon, beta=beta, batch_size=batch_size)
            replay_buffer.clear()
        if use_high_score_replay_buffer:
            if high_score_replay_buffer.has_enough_samples():
                for epoch in range(num_epochs_per_episode):
                    tennis_ppo_utils.run_training_epoch(agent, optimizer, high_score_replay_buffer,
                                                    discount=discount, epsilon=epsilon, beta=beta, batch_size=batch_size)
                if high_score_replay_buffer.sampled_count > high_score_buffer_sample_limit:
                    print("Clearing high score replay buffer")
                    high_score_replay_buffer.clear()

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
