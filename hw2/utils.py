import json

def state_action_dict(dataset):
    state_actions = {}
    for e in dataset:
        for r in e:
            state_actions[str(r['observation'])] = (
                    r['reward'],
                    r['action'],
                    r['next_observation'])

    return state_actions
