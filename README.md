# udacity-drlnd-tennis

# Introduction
This is my submission for the multi-agent tennis project for the Udacity Deep Reinforcement Learning nanodegree.  The purpose of the project is train 2 agents to bounce a ball over a net.

# Project Details
The environment for this project is a Unity ML-Agents environment provided by Udacity.  In this environment, each of 2 agents controls a tennis racket.  An agent receives a reward of +0.1 for hitting the ball over the net.  An agent receives a reward of -0.01 if it lets the ball hit the ground, or hits the ball out of bounds.  Therefore, the goal of each agent is to keep the ball in play.

The state space of each agent's observations is a vector of 8 variables corresponding to the position and velocity of the ball and racket.  Each agent receives its own local observation.  The environment provides the observations in stacks of 3, for a total length of 24 numbers.  Each agent's action space consists of 2 numbers between -1 and 1, corresponding to movement toward (or away) from the net, and jumping.

The score for an agent at the end of an episode is the sum of rewards that it received.  The score for an episode is considered to be the maximum of the agents' scores.  The environment is considered solved when the average over 100 episodes is at least 0.5.

# Getting Started
The dependencies for this submission are the same as for the [Udacity Deep Reinforcement Learning nanodegree](https://github.com/udacity/deep-reinforcement-learning#dependencies):
* Python 3.6
* pytorch
* unityagents
* [Udacity Tennis environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

This project was completed and tested on a local Windows environment (64-bit) running Python 3.6.10 in a conda environment.

# Instructions
To train the agent, run **train_tennis_agent.py**.  This will save the model parameters in **tennis_weights.pth** once the agent has fulfilled the criteria for considering the environment solved.

To run the trained agent, run **run_tennis_agent.py**.  This will load the saved model parameters from tennis_weights.pth, and run the trained model in the Tennis environment.  The *n_episodes* parameter is the number of episodes that will be run.  By default, this parameter is set to 100 to facilitate validation of the trained agent.

