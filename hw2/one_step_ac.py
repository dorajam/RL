import copy 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor():
    def __init__(self, state_actions):
        self.state_actions = state_actions

    def retrieve_action(self, state):
        reward, action, next_obs = self.state_actions[str(state)]
        return reward, action, next_obs


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, out_dim=1):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim+action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, state, action):
        # import ipdb;ipdb.set_trace()
        inp = torch.cat([state, action], 0)
        
        q = nn.ReLU()(self.l1(inp))
        q = nn.ReLU()(self.l2(q))
        q = self.l3(q)

        return q


class OneStepAC(object):
    def __init__(
            self,
            state_actions,
            state_dim=6,
            action_dim=3,
            gamma=0.99,
            eta=0.1,
            tau=0.005
    ):
        self.actor = Actor(state_actions)
        self.action_dim = action_dim
        self.critic = Critic(state_dim, action_dim)
        self.target_critic =  copy.deepcopy(self.critic)
        self.gamma = gamma
        self.tau = tau
        self.eta = eta
        self.criterion  = nn.MSELoss()
        self.critic_optimizer  = torch.optim.SGD(self.critic.parameters(), lr=self.eta)


    def update_target_parameters(self):
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


    def train_step(self, state):

        # Take action
        reward, action, next_state = self.actor.retrieve_action(state)
        # _, next_action, _ = self.actor.retrieve_action(next_state)


        # Get current Q estimates
        with torch.no_grad():
            action_vec = torch.zeros(self.action_dim)
            assert  action_vec.shape == (3,)
            action_vec[action] = 1
            state = torch.FloatTensor(state)
            target_val = self.target_critic(state, action_vec)
            target = torch.tensor(reward) + self.gamma * target_val  # pass in action?
        
        # next_action_vec = torch.zeros(self.action_dim)
        # next_action_vec[next_action] = 1
        next_state = torch.FloatTensor(next_state)

        next_value = self.critic(next_state, action_vec)
        td_error = self.criterion(next_value, target)
    
        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()

        self.update_target_parameters()
        if td_error > 5:
            import ipdb;ipdb.set_trace()
        return next_state, td_error
