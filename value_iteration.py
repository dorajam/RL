import pygame
import math
import numpy as np
from scipy.integrate import ode


##################
# VALUE ITERATION
##################


def reward(x, u, goal):
    from lqr_starter import Q, R
    # centers x around goal state
    x = x - goal

    return -np.dot(x, np.dot(Q, x)) + np.dot(u, np.dot(R, u))


def bellman_optimality_eq(state, actions, values, groups, goal, gamma):
    """
    state: current state idx
    actions: discretized intervals
    values: current sate values
    groups: discretized states 
    """
    action_values = bellman_eq(state, actions, values, groups, goal, gamma)
    best_action_idx = np.argmax(action_values)
    best_action = actions[best_action_idx]

    new_state_value = np.max(action_values)

    return best_action, new_state_value


def bellman_eq(state, actions, values, groups, goal, gamma=0.5):
    """
    state: discretized state
    actions: discretized actions
    values: state-value table
    """
    action_values = []

    # loop through each possible discrete action
    for action in actions:
        r = reward(state, action, goal)

        # takes discrete state and action, outputs continuous state
        cont_next_state = dynamics(state, action)

        # discretizes the state
        discrete_next_state, next_state_indices = discretize_state(groups, cont_next_state)

        next_value = values[tuple(next_state_indices)]


        curr = r + gamma * next_value
        action_values.append(curr)

    return action_values


def dynamics(state, action):
    # dynamics params
    g = 9.82
    m = 0.5
    M = 0.5
    l = 0.5
    b = 1.0

    z = state

    # continuous states
    f = np.array([action])

    sz = np.sin(z[3])
    cz = np.cos(z[3]) 
    cz2 = cz*cz

    a0 = m*l*z[2]*z[2]*sz
    a1 = g*sz
    a2 = f[0] - b*z[1]
    a3 = 4*(M+m) - 3*m*cz2

    dz = np.zeros((4,1))
    dz[0] = z[1]                                             # x
    dz[1] = (  2*a0 + 3*m*a1*cz + 4*a2 )/ ( a3 )             # dx/dt
    dz[2] = -3*( a0*cz + 2*( (M+m)*a1 + a2*cz ) )/( l*a3 )   # dtheta/dt
    dz[3] = z[2]                                             # theta

    return dz


def retrieve_state(x_id, x_prime_id, theta_prime_id, theta_id, groups):
    return np.array([
        groups[0][x_id],
        groups[1][x_prime_id],
        groups[2][theta_prime_id],
        groups[3][theta_id],
    ])


def value_iteration(states_table, actions, goal, groups, threshold=1e-4, gamma=0.1):
    """
    num_states (int): product of num_possible_states of each discretized state
    theta (float): threshold for stopping
    """
    num_x, num_x_prime, num_theta_prime, num_theta = states_table.shape
    
    values = np.random.rand(num_x, num_x_prime, num_theta_prime, num_theta)
    state_action = np.zeros((num_x, num_x_prime, num_theta_prime, num_theta))

    counter = 0 

    # optimal values
    while True:
        delta = 0.

        for x_id in range(num_x):
            for x_prime_id in range(num_x_prime):
                for theta_prime_id in range(num_theta_prime):
                    for theta_id in range(num_theta):

                        v = values[x_id, x_prime_id, theta_prime_id, theta_id]
                        state = retrieve_state(
                                x_id,
                                x_prime_id,
                                theta_prime_id,
                                theta_id,
                                groups
                        )

                        state_action[x_id, x_prime_id,theta_prime_id, theta_id],\
                        values[x_id, x_prime_id,theta_prime_id, theta_id] = bellman_optimality_eq(state, actions, values, groups, goal, gamma)

                        delta = max(delta, np.abs(v - values[x_id, x_prime_id,theta_prime_id, theta_id]))
        counter += 1
        print(delta)

        if delta < threshold:
            break


    return state_action


def compute_control_value_iter(policy, s):
    """
    policy (dict): state-action pairs
    s (np.array): state
    """
    return policy[s]


def discretize_state(groups, x):
        
    def get_lin_spaced_groups(state, group):
        dist = np.abs(np.tile(state, group.shape[0]) - group)
        idx = dist.argmin()
        return idx

    discrete_x = np.zeros_like(x)
    discrete_indices = []

    # position groups: x<-3, x in [-3,3], x>3
    idx = get_lin_spaced_groups(x[0], groups[0])
    discrete_x[0] = groups[0][idx]
    discrete_indices.append(idx)

    # velocity: 40 lin spaced groups
    idx = get_lin_spaced_groups(x[1], groups[1])
    discrete_x[1] = groups[1][idx]
    discrete_indices.append(idx)

    # angle velocity
    idx = get_lin_spaced_groups(x[2], groups[2])
    discrete_x[2] = groups[2][idx]
    discrete_indices.append(idx)

    # angle
    idx = get_lin_spaced_groups(x[3], groups[3])
    discrete_x[3] = groups[3][idx]
    discrete_indices.append(idx)

    return discrete_x, discrete_indices



def discretize_state_space():
    # position
    group0 = np.array([-2, 0, 2])

    # velocity 
    group1 = np.linspace(-.5, .5, 5)

    # angle velocity
    group2 = np.linspace(-7,7,20)

    # angle
    group3 = np.append(np.array([0]), np.linspace(1.57, 4.71, 10))
    group3 = np.append(group3, np.asarray([6.28]))

    groups = [group0, group1, group2, group3]

    goal, _ = discretize_state(groups, np.array([0,0,0, 3.14]))
    return groups, goal
