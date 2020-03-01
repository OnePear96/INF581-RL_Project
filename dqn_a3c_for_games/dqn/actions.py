import math
import torch
import random
import gym

'''
LEFT = [-1.0, 0.0, 0.0]
RIGHT = [1.0, 0.0, 0.0]
GAS = [0.0, 1.0, 0.0]
BRAKE = [0.0, 0.0, 1.0]

ACTIONS = [LEFT, RIGHT, GAS, BRAKE]


def get_action_space():
    return len(ACTIONS)
'''

game = "Breakout-ram-v0"
env = gym.make(game)
ACTIONS = env.action_space



def get_action_space():
    return ACTIONS.n


def get_action(q_value, train=False, step=None, params=None, device=None):
    if train:
        epsilon = params.epsilon_final + (params.epsilon_start - params.epsilon_final) * \
            math.exp(-1 * step / params.epsilon_step)
        if random.random() <= epsilon:
            action_index = random.randrange(get_action_space())
            action = action_index
            return torch.tensor([action_index], device=device)[0], action
    action_index = q_value.max(1)[1]
    action = action_index[0]
    return action_index[0], action
