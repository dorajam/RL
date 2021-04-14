import torch


def state_action_dict(dataset):
    state_actions = {}
    for e in dataset:
        for r in e:
            state_actions[str(r['observation'])] = (
                r['reward'],
                r['action'],
                r['next_observation'])

    return state_actions


def split_data(X, y, ratio=(0.6, 0.3)):
    assert sum(ratio) < 1., 'Ratio arg invalid. You need to define a split to allow for test samples.'

    num_samples = X.shape[0]
    indices = torch.randperm(X.shape[0])
    tr_idx = int(num_samples * ratio[0])
    dev_idx = int(num_samples * ratio[1])
    print(f'Training id and dev id: {tr_idx}, {dev_idx}')

    training_X = X[indices[:tr_idx]]
    training_y = y[indices[:tr_idx]]
    dev_X = X[indices[tr_idx:tr_idx + dev_idx]]
    dev_y = y[indices[tr_idx:tr_idx + dev_idx]]
    test_X = X[indices[tr_idx + dev_idx:]]
    test_y = y[indices[tr_idx + dev_idx:]]

    print(f'Created {len(training_X)} training, {len(dev_X)}, dev and {len(test_X)} test samples')

    return (
        (training_X, training_y),
        (dev_X, dev_y),
        (test_X, test_y)
    )
