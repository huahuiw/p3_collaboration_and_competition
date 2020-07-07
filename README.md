# p3_collaboration_and_competition

## Introduction
In this project, two agents collaborate with each other to play the ball. A reward of +0.1 is provided each time the ball is hit over the net and a penalty of -0.01 when the ball hits the ground. Thus, the goal of the agents is to hit the ball over the net for as many times as possible.

The observation space for each agent consists of 24 variables. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. Each action is a vector with two numbers. Every entry in the action vector should be a number between -1 and 1.

The problem is considered solved when the average reward exceeds 0.5 over 100 episodes, after taking the maximum over both agents.

## Setup Instructions
Follow the instructions in the following link to set up the environment.
https://github.com/udacity/deep-reinforcement-learning#dependencies

Download the Unity Environment (Windows 64bit) and place the unzip files in the same directory of the remaining files of the same project.
https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip

## How to run the codes
Activate drlnd environment.

Navigagte to the directory of p3_collaboration_and_competition where all the files are together.
Use Jupyter notebook to open report.ipynb.

Follow the instructions in that notebook step by step to train the agent. First start the environment, and then you can train the agent in Section 2 by importing necessary modules.

You can also jump directly to Section 3 'Testing' (after starting the environment in Section 1), to load the saved weights and use the trained agent to play the game. The trained weights are stored in file 'checkpoint_actor_agent{x}.pth' and 'checkpoint_critic_agent{x}.pth' with x= 0 or 1, corresponding to weights of actor and critic networks for agent x, respectively. 
