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
    
class VWLinearReward(mdp.reward.LinearReward):
    '''
    Equivalent to VWReward, but specified in a linear fashion
    '''
    def __init__(self, map):
        self._map = map
        weights = -1*np.ones( self.dim )
        weights[-1] = 0
        self.params = weights
        
    def features(self, state, action):
        result = np.zeros( self.dim )
        dust = state.dust
        for (i,idx) in enumerate( np.transpose( np.nonzero(np.ones( self._map.shape )) ) ):
            result[i] = dust[tuple(idx)]
        if isinstance(action, vacuumworld.model.VWMoveAction):
            result[-2] = 1
        elif isinstance(action, vacuumworld.model.VWSuckAction):
            result[-1] = 1 
        return result
    
    @property
    def dim(self):
        return np.prod(self._map.shape) + 2