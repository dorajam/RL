import gym
import math
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import json
import argparse


class PolicyNetwork(nn.Module):

    def __init__(self, epsilon=0.0, num_states=6, num_actions=3, 
            hidden_dim=10):
        super(PolicyNetwork, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.mlp = nn.Sequential(
            nn.Linear(self.num_states, hidden_dim),
            nn.Linear(hidden_dim, self.num_actions)
        )
        self.epsilon = epsilon

    def forward(self, state):
        state = torch.Tensor(state)
        out = self.mlp(state)
        action = nn.Softmax(-1)(out)
        action = torch.argmax(action)
        # take random action with epsilon probability
        flip = torch.rand(1)
        if flip > 1 - self.epsilon:
            action = torch.randint(0, 3, (1,))
        return action 


def normalize(obs, means, stdevs):
    return (obs - means) / stdevs


def run_env(network, render=False, episodes=500, timestamps=100, means=None, stdevs=None):
    res = []
    data = []
    env = gym.make('Acrobot-v1')

    for episode in range(episodes):
        episode_data = []

        observation = env.reset()
        if episode%50 == 0:
            print(f'Running episode {episode}...')

        for t in range(timestamps):
            if render:
                env.render()
            # normalize states
            if np.any(means) and np.any(stdevs):
                observation = normalize(observation, means, stdevs)

            if episode_data:
                episode_data[-1]['next_observation'] = observation.tolist()

            # forward prop -> get action
            action = network(observation)
            res.append(observation)

            # takes action
            out = env.step(action)
            next_observation, reward, done, info = out

            episode_data.append({
                'observation': observation.tolist(),
                'next_observation': None,
                'reward': reward,
                'action': action.item(),
                'done': done
            })
            observation = next_observation

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break

        data.append(episode_data)

    if render: env.close()
    return res, data

def main(args):
    torch.manual_seed(0)
    network = PolicyNetwork(args.eps)

    # collect avgs and stdevs
    states, _ = run_env(network, render=False, episodes=500, timestamps=50)
    means = np.mean(states, axis=0)
    stdevs = np.std(states, axis=0)
    # run policy
    _, data = run_env(network, render=False, episodes=500, timestamps=500, means=means, stdevs=stdevs)
    with open(f'dataset_eps{args.eps}.jsonl', 'w+') as f:
        for obs in data:
            f.write(json.dumps(obs) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-eps", "--eps", "-e", default=0.0, type=float)
    args = parser.parse_args()
    main(args)
