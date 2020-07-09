import numpy as np
import torch
from ddpg import DDPG_agent
from collections import namedtuple, deque
import random

class MADDPG:
    def __init__(self, num_agents, state_size=24, action_size=2, buffer_size = int(1e6), batch_size = 256, lr_actor=0.001, lr_critic=0.001, sigma=0.0):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.agents = []
        for i in range(self.num_agents):
            agent = DDPG_agent(state_size, action_size, num_agents*state_size, num_agents*action_size, buffer_size, batch_size, lr_actor, lr_critic, sigma)
            self.agents.append(agent)
        
        self.memory = ReplayBuffer(buffer_size, batch_size)
        
    def update_noise(self, new_sigma):
        for i in range(self.num_agents):
            agent = self.agents[i]
            agent.noise.update_sigma(new_sigma)
            
    def act(self, states, training = True):
        actions = []
        for i in range(self.num_agents):
            action = self.agents[i].act(states[i], training = training)
            actions.append(action)
        actions_full = np.array(actions)
        #print("actions_full shape", actions_full.shape)
        return actions_full ## np array of shape: num_agents * action_size
    
    def step(self, states, actions, rewards, next_states, dones):

        #print("states shape:", states.shape)
        #print("next_states shape:", next_states.shape)
        states_full = np.reshape(states, -1)
        next_states_full = np.reshape(next_states, -1)
        
        self.memory.add(states, states_full, actions, rewards, next_states, next_states_full, dones)
        
        if len(self.memory) > 10*self.batch_size:
            for i in range(self.num_agents):
                Experiences = self.memory.sample()
                self.learn(i,Experiences)
            self.agent_update()
    
    def agent_update(self):
        for i in range(self.num_agents):
            agent = self.agents[i]
            agent.update_all()
        
   
    def learn(self, agent_n, Experience):
        states, states_full, actions, rewards, next_states, next_states_full, dones = Experience
        
      
        ## generate full actions for both agents to train their local critic and actor networks 
        actions_next_full = []
        for i in range(self.num_agents):
            agent = self.agents[i]
            #print("next_states shape:",next_states.shape)
            actions_next = agent.actor_target(next_states[:,i,:])
            actions_next_full.append(actions_next)
            
        actions_next_full = torch.cat(actions_next_full, dim = 1)
        #print("actions_next_full size", actions_next_full.size())
        
        
        ## generate full actions using states from a given agent's actor_local network
        agent = self.agents[agent_n]
        action_pred = agent.actor_local(states[:,agent_n,:])
        
        #print("action_pred size", action_pred.size())
        
        full_actions_pred = actions.clone()
        full_actions_pred[:,agent_n,:] = action_pred
        full_actions_pred = full_actions_pred.view(-1,self.num_agents*self.action_size)
        #print("full_actions_pred size:",full_actions_pred.size())
        
        actions_full = actions.view(-1,self.num_agents*self.action_size)
        
        local_rewards = rewards[:,agent_n]
        local_dones = dones[:,agent_n]
        ## use overall rewards instead of local rewards
        #local_rewards = torch.max(rewards, dim=1)
        
        
        local_rewards = torch.unsqueeze(local_rewards, dim=1)
        local_dones = torch.unsqueeze(local_dones, dim=1)

        experience = (states_full,  actions_full, local_rewards, next_states_full, local_dones, full_actions_pred, actions_next_full)
        agent.learn(experience)
        
    def save_weights(self):
        for i in range(self.num_agents):
            agent = self.agents[i]
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor'+'_agent'+str(i)+'.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic'+'_agent'+str(i)+'.pth')

    def load_weights(self):
        for i in range(self.num_agents):
            self.agents[i].actor_local.load_state_dict(torch.load('checkpoint_actor'+'_agent'+str(i)+'.pth'))
            self.agents[i].critic_local.load_state_dict(torch.load('checkpoint_critic'+'_agent'+str(i)+'.pth'))
             
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.batch_size = batch_size
        self.buffer = deque(maxlen = buffer_size)
        self.Experience = namedtuple('Experience', 'states states_full actions rewards next_states next_states_full dones')

    def add(self, states, states_full, actions, rewards, next_states, next_states_full, dones):
        
       # print("states shape:", states.shape)
        e = self.Experience(states, states_full, actions, rewards, next_states, next_states_full, dones)
        self.buffer.append(e)

    def sample(self):
        experiences = random.sample(self.buffer, self.batch_size)
        #print("experience shape", experiences[0].states.shape)
        
        states = torch.from_numpy(np.array([e.states for e in experiences if e is not None])).float()
        #print("states size:", states.size())
        states_full = torch.from_numpy(np.array([e.states_full for e in experiences if e is not None])).float()
        #print("states_full size:", states_full.size())
        actions = torch.from_numpy(np.array([e.actions for e in experiences if e is not None])).float()
        #print("actions size:", actions.size())
        rewards = torch.from_numpy(np.array([e.rewards for e in experiences if e is not None])).float()
        #print("rewards size:", rewards.size())
        next_states = torch.from_numpy(np.array([e.next_states for e in experiences if e is not None])).float()
        #print("next_states size:", next_states.size())
        next_states_full = torch.from_numpy(np.array([e.next_states_full for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.array([e.dones for e in experiences if e is not None])).float()
        return (states, states_full, actions, rewards, next_states, next_states_full, dones)

    def __len__(self):
        return len(self.buffer)
    