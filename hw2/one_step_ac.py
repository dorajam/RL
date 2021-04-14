import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor():
    def __init__(self, state_actions, action_dim):
        self.state_actions = state_actions
        self.action_dim = action_dim

    def retrieve_action(self, state, device):
        try:
            reward, action, next_state = self.state_actions[str(state)]

            # convert types
            action_vec = torch.zeros(self.action_dim).to(device)
            assert action_vec.shape == (3,)
            action_vec[action] = 1.
            state = torch.FloatTensor(state).to(device)
            reward = torch.tensor([reward]).to(device)

            return state, reward, action_vec, next_state
        except:
            return None, None, None, None


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=400, hidden_dim2=300, out_dim=1):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim1)
        self.l2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.l3 = nn.Linear(hidden_dim2, out_dim)

    def forward(self, state, action):
        inp = torch.cat([state, action], 1)

        q = nn.ReLU()(self.l1(inp))
        q = nn.ReLU()(self.l2(q))
        q = self.l3(q)

        return q


class ExperienceReplay(object):
    def __init__(self, state_dim=6, action_dim=3, max_size=100, device='cuda'):
        self.max_size = max_size
        self.states = torch.zeros((state_dim, max_size))
        self.actions = torch.zeros((action_dim, max_size))
        self.rewards = torch.zeros((1, max_size))
        self.next_states = torch.zeros((state_dim, max_size))
        self.next_actions = torch.zeros((action_dim, max_size))
        self.not_done_flag = torch.ones((1, max_size))
        self.size = 0
        self.pointer = 0
        self.device = device

    def add(self, new):
        not_done = 1.
        state, action, reward, next_state, next_action = new

        if not isinstance(next_state, torch.Tensor):
            not_done = 0.
            next_state = torch.zeros(state.shape)
            next_action = torch.zeros(action.shape)

        self.states[:, self.pointer] = state
        self.actions[:, self.pointer] = action
        self.rewards[:, self.pointer] = reward
        self.next_states[:, self.pointer] = next_state
        self.next_actions[:, self.pointer] = next_action
        self.not_done_flag[:, self.pointer] = not_done

        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        pick_ids = torch.randperm(self.size)[:min(self.size, batch_size)]

        return (
            torch.FloatTensor(self.states[:, pick_ids]).transpose(0, 1).to(self.device),
            torch.FloatTensor(self.actions[:, pick_ids]).transpose(0, 1).to(self.device),
            torch.FloatTensor(self.rewards[:, pick_ids]).transpose(0, 1).to(self.device),
            torch.FloatTensor(self.next_states[:, pick_ids]).transpose(0, 1).to(self.device),
            torch.FloatTensor(self.next_actions[:, pick_ids]).transpose(0, 1).to(self.device),
            torch.FloatTensor(self.not_done_flag[:, pick_ids]).transpose(0, 1).to(self.device),
        )


class DDPG(nn.Module):
    def __init__(
            self,
            state_actions,
            state_dim=6,
            action_dim=3,
            gamma=0.99,
            eta=1e-2,
            tau=0.001,
            max_size=100,
            device='cuda'
    ):
        super(DDPG, self).__init__()
        self.actor = Actor(state_actions, action_dim)
        self.action_dim = action_dim
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.gamma = gamma
        self.tau = tau
        self.eta = eta
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.eta)
        self.experience_replay = ExperienceReplay(max_size=max_size, device=device)
        self.device=device

    def update_target_parameters(self):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def train_step(self, state, batch_size):
        # Take action
        state, reward, action, next_state = self.actor.retrieve_action(state, self.device)
        _, _, next_action, _ = self.actor.retrieve_action(next_state, self.device)

        try:
            next_state = torch.FloatTensor(next_state)
        except:
            next_state = None

        # Experience replay
        new_row = [state, action, reward, next_state, next_action]

        self.experience_replay.add(new_row)

        state_b, a_b, r_b, ns_b, na_b, not_done = \
            self.experience_replay.sample(batch_size)

        target_q = self.target_critic(ns_b, na_b)
        target_q = (r_b + not_done * self.gamma * target_q).detach()

        current_q = self.critic(state_b, a_b)
        td_error = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()

        self.update_target_parameters()
        return td_error

    def eval_step(self, state):
        # Take action
        state, reward, action, next_state = self.actor.retrieve_action(state, device=self.device)
        _, _, next_action, _ = self.actor.retrieve_action(next_state, device=self.device)

        try:
            next_state = torch.FloatTensor(next_state).to(self.device)
        except:
            next_state = None
        # import ipdb;
        # ipdb.set_trace()

        if not isinstance(next_state, torch.Tensor):
            target_q = reward.reshape(1,-1)
        else:
            target_q = self.target_critic(next_state.reshape(1,-1), next_action.reshape(1,-1))
            target_q = (reward + self.gamma * target_q).detach()

        current_q = self.critic(state.reshape(1,-1), action.reshape(1,-1))
        td_error = F.mse_loss(current_q, target_q)

        return td_error, current_q, reward

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.target_critic = copy.deepcopy(self.critic)
