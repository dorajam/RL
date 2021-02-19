import pygame
import math
import numpy as np
from scipy.integrate import ode
from lqr_starter import Q, R


##################
# VALUE ITERATION
##################


def reward(Q, R, x, u):
    return np.dot(x, np.dot(Q, x)) + np.dot(u, np.dot(R, u))


def bellman_optimality_eq(state, reward, actions, value_next_states):
    return max(bellman_eq(state, reward, actions, value_next_states))


def bellman_eq(state, reward, actions, value_next_states):
    new_values = []
    for action, v_next in zip(actions, value_next_states):
        curr = sum(dynamics(state, action) * reward + gamma * v_next)
        new_values.append(curr)
    return new_values


def retrieve_state(x_id, x_prime_id, theta_prime_id, theta_id, groups):
    return np.array([
        groups[0][x_id],
        groups[2][x_prime_id],
        groups[3][theta_prime_id],
        groups[4][theta_id],
    ])


def value_iteration(states_table, goal_state_indices, groups, theta=1e-4):
    """
    num_states (int): product of num_possible_states of each discretized state
    theta (float): threshold for stopping
    """
    num_states = states_table.shape
    
    values = np.random.rand(states_table.shape)  # ordered according to transitions
    values[goal_state_indices] = 0   # terminal state
    # values = values.reshape(-1)
    
    # optimal values
    while True:
        delta = 0.

        for x_id in num_states[0]:
            for x_prime_id in num_states[1]:
                for theta_prime_id in num_states[2]:
                    for theta_id in num_states[3]:

                        v = values[x_id, x_prime_id, theta_prime_id, theta_id]
                        state = retrieve_state(
                                x_id,
                                x_prime_id,
                                theta_prime_id,
                                theta_id,
                                groups
                        )
                        # TODO
                        next_state = 
                        r = reward(Q, R, x, u)
                        actions = None
                        next_state_value = values[next_state]

                        values[state] = bellman_optimality_eq(
                                state, r, actions, next_state_values)

                        delta = max(delta, np.abs(v - values[s]))

                        if delta < theta:
                            break

    # new policy: state to action idx
    policy = {s: None for s in states}
    for s in policy.keys():
        next_states = None
        next_state_values = values[next_states]
        policy[s] = np.argmax(bellman_eq(state, reward, actions, value_next_states))

    return policy


def compute_control_value_iter(policy, s):
    """
    policy (dict): state-action pairs
    s (np.array): state
    """
    return policy[s]


def discretize_states(groups, x):
        
    def get_lin_spaced_groups(state, group):
        dist = np.abs(np.tile(state) - group)
        idx = dist.argmin()
        return idx

    discrete_x = np.zeros_like(x)

    # position groups: x<-3, x in [-3,3], x>3
    idx = get_lin_spaced_groups(x[0], groups[0])
    discrete_x[0] = groups[ix]

    # velocity: 40 lin spaced groups
    idx = get_lin_spaced_groups(x[1], groups[1])
    discrete_x[1] = groups[ix]

    # angle velocity
    idx = get_lin_spaced_groups(x[2], groups[2])
    discrete_x[2] = groups[ix]

    # angle
    idx = get_lin_spaced_groups(x[3], groups[3])
    discrete_x[3] = groups[ix]

    return discrete_x


def discretize_state_space():
    # position groups: x<-3, x in [-3,3], x>3
    group0 = np.array([-6, 0, 6])

    # velocity: 40 lin spaced groups
    group1 = np.linspace(-20,20,41)

    # angle velocity
    group2 = np.linspace(-20,20,41)

    # angle
    group3 = np.linspace(0,9.42,52)

    groups = [group0, group1, group2, group3]

    goal_state_indices = [1, 20, 20, 17]
    return groups, goal_state_indices


groups, goal_state_indices = discretize_state_space()
num_states = list(map(lambda r: len(r), groups))
states_table = np.zeros(num_states)
