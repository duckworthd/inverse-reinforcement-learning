'''
Created on Mar 31, 2011

@author: duckworthd
'''
from numpy import dot

class Reward(object):
    '''
    A Reward function stub
    '''

    def __init__(self):
        self._params = []
    
    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self,_params):
        self._params = _params
        
    def reward(self, state, action):
        raise NotImplementedError()
    
class LinearReward(Reward):
    '''
    A Linear Reward function stub
    
    params: weight vector equivalent to self.dim()
    '''
    def features(self, state, action):
        raise NotImplementedError()
    
    @property
    def dim(self):
        raise NotImplementedError()
    
    def reward(self, state, action):
        return dot(self.params, self.features(state,action))