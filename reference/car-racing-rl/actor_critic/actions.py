import torch
import numpy as np


LEFT = [-0.8, 0.0, 0.0]
RIGHT = [0.8, 0.0, 0.0]
GAS = [0.0, 0.5, 0.0]
BRAKE = [0.0, 0.0, 0.8]

ACTIONS = [LEFT, RIGHT, GAS, BRAKE]


def get_action_space():
    return len(ACTIONS)


def get_actions(probs):
 #   print("probs:",probs)
    values, indices = probs.max(1)
    actions = np.zeros((probs.size(0), 3))
    for i in range(probs.size(0)):
        action = ACTIONS[indices[i]]
        actions[i] = np.array(action)  * float(values[i])
#    print ("actions:",actions)
    return actions
