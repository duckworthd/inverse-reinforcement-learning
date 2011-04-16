'''
Created on Apr 16, 2011

@author: duckworthd
'''
import mdp.reward
import vacuumworld.model
import numpy as np

class VWReward(mdp.reward.Reward):
    '''
    A Reward function that penalizes the number of 
    dusty locations and the action chosen.
    '''
        
    def reward(self, state, action):
        result = -1 * np.sum(state.dust)
        if isinstance(action, vacuumworld.model.VWMoveAction):
            result -= 1
        elif isinstance(action, vacuumworld.model.VWSuckAction):
            result -= 0
        return result