import gym
import math
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import json
import argparse
from one_step_ac import OneStepAC
import utils


def train(model, dataset, epochs=100):
    episodes = len(dataset)
    epoch_td_error = []

    for e in range(epochs):

        td_errors = []
        # loop through episodes
        for episode in range(episodes):

            if episode%50 == 0:
                print(f'Running episode {episode}...')

            if episode+1%5 == 0:
                if td_errors:
                    print(f'TD error at episode {episode}: ', td_errors[-1])

            # loop through episode's trajectory
            for sample in dataset[episode]:
                import ipdb;ipdb.set_trace()
                observation = sample['observation']
                next_state, error = model.train_step(observation)

                if torch.isnan(error):
                    import ipdb;ipdb.set_trace()
                    
                td_errors.append(error.item())

        epoch_td_error.append(np.mean(td_errors))
            
    return td_errors


def main(args):
    torch.manual_seed(0)
    dataset = []
    with open(args.dataset, 'r') as f:        
        for d in f.readlines():
            dataset.append(json.loads(d))

    state_actions = utils.state_action_dict(dataset)
    model = OneStepAC(state_actions=state_actions)

    # run model
    td_errors = train(model, dataset, args.epochs)
    
    # save losses
    # with open(f'errors_{args.dataset}.txt', 'w+') as f:
        # f.write(json.dumps(td_errors))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "--dataset", "-d", default='dataset_eps0.0.txt', type=str)
    parser.add_argument("-epochs", "--epochs", "-e", default=100, type=int)
    args = parser.parse_args()
    main(args)
