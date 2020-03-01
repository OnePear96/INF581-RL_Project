import torch
import numpy as np
import gym
import sys 

#from params import Params
'''

LEFT = [-0.5, 0.0, 0.0]
RIGHT = [0.5, 0.0, 0.0]
GAS = [0.0, 0.8, 0.0]
BRAKE = [0.0, 0.0, 1.0]
ACTIONS = [LEFT, RIGHT, GAS, BRAKE]
'''
#params = Params('params/a3c.json')



game = "Atlantis-v0"
env = gym.make(game)
ACTIONS = env.action_space



def get_action_space():
    return ACTIONS.n
   #  return len(ACTIONS)
 
def get_actions(probs):
    values, indices = probs.max(1)
    '''
    actions = np.zeros((probs.size(0), 3))
    for i in range(probs.size(0)):
        action = indices[i]
        actions[i] = np.array(action)
    print ("actions:",actions)
    print("probs:",probs)
    '''
    return indices
 #   return np.array(indices)

'''
def get_actions(probs):
    values, indices = probs.max(1)
    actions = np.zeros((probs.size(0), 3))
    for i in range(probs.size(0)):
        action = ACTIONS[indices[i]]
        actions[i] = float(values[i]) * np.array(action)
    return actions
'''

if __name__ == '__main__':
    print(get_action_space())
    probs = [0.1,0.2,0.3,0.4]
    get_actions(probs)