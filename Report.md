# Project Report
This document is my report for the Tennis project.

# Learning Algorithm
This learning algorithm is an adaptation of Proximal Policy Optimization (PPO), using an Actor-Critic architecture and Generalized Advantage Estimation (GAE).  I modeled the code implementation after the OpenAI SpinningUp example of PPO with PyTorch:
https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo

The Actor-Critic architecture allows the agent to learn from both a policy-based neural network (the Actor), as well as value-based neural network (the Critic).  The Actor learns to take actions based on observations of the environment state, and the Critic learns to predict the state values of the observed state.  In this way, the Critic's predictions are used to decrease variance while training the Actor.

For this implementation, the Actor uses a neural network architecture with the following layers and attributes, coded in TennisActorCritic.py:
* A Linear layer with input size 24 (to match state space) and output size 64
* A Linear layer with input size 64 and output size 64
* A Linear layer with input size 64 and output size 2 (to match action space)
* Leaky ReLU is used as the activation function for the first 2 layers, to bring the benefits of ReLU while allowing some learning for values below zero.
* Hyperbolic tangent (tanh) is used as the activation function for the last layer, to output values between -1 and 1.
* The Actor network outputs the means of a Normal distribution for each action, with the standard deviations being neural network parameters.
* The actions to take are then determined by sampling from the Normal distributions.

The Critic uses a neural network architecture with the following layers and attributes, coded in TennisActorCritic.py:
* A Linear layer with input size 24 (to match state space) and output size 64
* A Linear layer with input size 64 and output size 64
* A Linear layer with input size 64 and output size 1 (to output a scalar state value)
* Leaky ReLU is used as the activation function for the first 2 layers, to bring the benefits of ReLU while allowing some learning for values below zero.
* No activation function is used for the last layer, to allow the Critic to output any value for the state value.

The Actor and Critic are implemented as separate networks with separate optimizers.  The same agent is used for both players.  This works because each player has its own local observation, and this allows for experiences from both players to be stored in the same buffer for learning.  The experience buffer, coded in TrajectoryBuffer.py, is used to collect experiences from both players for training.  Playouts are run until the buffer is full, at which point a round of training is performed, and the buffer experiences discarded (since PPO is an on-policy algorithm).

The following hyperparameters are used in this project:
* PPO buffer size: the number of samples stored in the TrajectoryBuffer before a round of training is performed
* Actor learning rate: learning rate for the Actor network
* Critic learning rate: learning rate for the Critic network
* Clip ratio: This is the clip ratio for PPO (represented with the Greek letter epsilon in some papers).  
* Entropy coefficient: Entropy coefficient used in the Actor loss function, to encourage exploration at the beginning.  This value is set to decay as training progresses so that there is less exploration once the Actor has learned a good policy.
* Number of iterations to train Actor and Critic: These are separate values to control how many gradient descent steps are taken at each round of training.

Interestingly, from experimentation, I found that the PPO buffer size and entropy coefficient are very significant hyperparameters.  In particular, increasing buffer size from 128 to 1024, and the entropy coefficient from 0.01 to 0.02 made huge improvements in performance of the PPO algorithm.

# Reward Plots
The agent solved the environment in 9041 episodes.  A plot of the training rewards is shown below.

For further validation, the trained agent was run through the environment for an additional 100 episodes, achieving an average score of XXX.  A plot of the validation rewards is shown below.

# Ideas for Future Work
I solved this project with PPO, using an Actor-Critic architecture and Generalized Advantage Estimation (GAE).  The following are ideas for future work:
* The Actor-Critic network could be reduced in size by combining the initial layers of the Actor and Critic.  This would allow the Actor and Critic to learn from a shared understanding of the environment.
* This PPO implementation is less sample-efficient than the benchmark implementation shown in the Udacity introduction for the project, which uses Multi-Agent Deep Deterministic Policy Gradient (MA-DDPG).
* Some of the more recent reinforcement learning algorithms, such as Soft Actor-Critic (SAC) and Twin Delayed DDPG (TD3), would be interesting to apply for this project.  I think SAC is interesting in particular because I found the entropy coefficient to be important in this implementation, and SAC utilizes entropy regularization as a key part of the algorithm.

# Acknowledgments
As previously mentioned, I would like to acknowledge the OpenAI SpinningUp example of PPO, which I used as a resource to guide my implementation:
https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo
