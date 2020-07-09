import numpy as np
import torch
from model import Actor, Critic
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import copy

TAU = 0.001
GAMMA = 0.99

class DDPG_agent:
    def __init__(self, actor_state_size, actor_action_size, critic_state_size, critic_action_size, buffer_size, batch_size, lr_actor, lr_critic, sigma=0.0):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.sigma=sigma
        self.actor_local = Actor(actor_state_size, actor_action_size)
        self.actor_target = Actor(actor_state_size, actor_action_size)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        self.critic_local = Critic(critic_state_size, critic_action_size)
        self.critic_target = Critic(critic_state_size, critic_action_size)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=0.)

        self.noise = Noise(size=actor_action_size, mu=0., sigma=self.sigma)
        

    def act(self, states, training=True):
        states = torch.from_numpy(states).float()
        #actions = self.actor_local.forward(states)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if training:
            actions = actions + self.noise.sample()
        return np.clip(actions, -1, 1)

    def learn(self, Experience):
        states_full,  actions_full, rewards, next_states_full, dones, full_actions_pred, actions_next_full = Experience
        ## update critic

        #print("next_states_full size:", next_states_full.size())
        #print("actions_next_full size:", next_states_full.size())
        
        
        Q_next = self.critic_target(next_states_full, actions_next_full)
        #print("Q_next size:", Q_next.size())
        
        #rewards = torch.unsqueeze(rewards, dim=1)
        #dones = torch.unsqueeze(dones, dim=1)
        #print("rewards size:", rewards.size())
        

        Q_target = rewards + GAMMA * Q_next * (1 - dones)
    
        Q_expect = self.critic_local(states_full, actions_full)

        critic_loss = F.mse_loss(Q_target, Q_expect)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        ## update actor

        Q = self.critic_local(states_full, full_actions_pred)
        actor_loss = -Q.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    ## update target networks
    def update_all(self):
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for local, target in zip(local_model.parameters(), target_model.parameters()):
            target.data.copy_(tau * local.data + (1 - tau) * target.data)


# class ONNoise:
#     def __init__(self, size, theta, mu, sigma):
#         self.size = size
#         self.theta = theta
#         self.mu = np.ones(size) * mu
#         self.sigma = sigma
#         self.x = copy.copy(self.mu)

#     def sample(self):
#         dx = self.theta * (self.mu - self.x) + self.sigma * np.random.random(size=self.size)
#         self.x = self.x + dx
#         return self.x

class Noise:
    def __init__(self, size, mu, sigma):
        self.size = size
        self.sigma = sigma
        self.mu = mu
        self.x = copy.copy(np.ones(size)*mu)
        
    def update_sigma(self, new_sigma):
        self.sigma = new_sigma
        self.x = np.random.normal(self.mu, self.sigma, self.size)
        return self.x
    
    def sample(self):
        self.x = np.random.normal(self.mu, self.sigma, self.size)
        return self.x

    
