import argparse
import json
import os

import numpy as np
import torch
import utils
from ddpg import DDPG


def train(model, dataset, epochs=100, save_path=''):
    episodes = len(dataset)
    epoch_td_error = []

    for e in range(epochs):
        iter = 0
        episode_errors = []

        # loop through episodes
        for episode in range(episodes):

            if episode % 200 == 0:
                print(f'Running episode {episode}...')
            if episode % 50 == 0:
                if episode_errors:
                    print(f'Running episode {episode}...')
                    print(f'AVG TD error at episode {episode}: ', np.sum(episode_errors)/iter)

            # if episode_errors:
            # print(f'Avg episode TD error at episode {episode}: ', episode_errors[-1])

            td_errors = 0

            # loop through episode's trajectory
            for sample in dataset[episode]:
                observation = sample['observation']
                error = model.train_step(observation, batch_size=128)
                td_errors += error.item()
                iter += 1

            episode_errors.append(td_errors)

        epoch_error = np.sum(episode_errors) / iter
        epoch_td_error.append(epoch_error)
        print(f'TD error at epoch {e}: ', epoch_error)

        # save model
        if args.save_path:
            model.save(f"{save_path}/model_at_epoch_{e}")


    return epoch_td_error


def eval(model, dataset, episode=100):
    td_errors = []
    predicted_rewards = []
    rewards = []
    # loop through episode's trajectory
    for sample in dataset[episode]:
        observation = sample['observation']
        td_error, current_q, reward = model.eval_step(observation)

        td_errors.append(td_error.item())
        predicted_rewards.append(current_q.item())
        rewards.append(reward.item())

    return td_errors, predicted_rewards, rewards


def main(args):
    torch.manual_seed(0)
    dataset = []
    with open(args.dataset, 'r') as f:
        for d in f.readlines():
            dataset.append(json.loads(d))

    state_actions = utils.state_action_dict(dataset)
    model = DDPG(state_actions=state_actions, device=args.device)
    if args.load_model:
        model.load(os.path.join(args.save_path, args.load_model))

    if not args.eval_only:
        # run model
        epoch_errors = train(model, dataset, args.epochs, save_path=args.save_path)

        # save losses
        with open(f'epoch_errors_{args.dataset[:-6]}.txt', 'w+') as f:
            f.write(json.dumps(epoch_errors))

    errors, predicted, rewards = eval(model, dataset)

    with open(f'predicted_rewards_{args.dataset[:-6]}.txt', 'w+') as f:
        for e in predicted:
            f.write(json.dumps(e) + '\n')

    with open(f'actual_rewards_{args.dataset[:-6]}.txt', 'w+') as f:
        for r in rewards:
            f.write(json.dumps(r) + '\n')

    with open(f'eval_errors_{args.dataset[:-6]}.txt', 'w+') as f:
        for err in  errors:
            f.write(json.dumps(err) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "--dataset", "-d", default='dataset_eps0.0.jsonl', type=str)
    parser.add_argument("-epochs", "--epochs", "-e", default=10, type=int)
    parser.add_argument("--save_path", default='./models')
    parser.add_argument("--eval_only", default=False)
    parser.add_argument("--load_model", default="", help='Filename to load from')
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.save_path = args.save_path + '/' + args.dataset[:-6]
    if not os.path.exists(args.save_path):
        print(f'Creating {args.save_path}')
        os.makedirs(args.save_path)

    main(args)
