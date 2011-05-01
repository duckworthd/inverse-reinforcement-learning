'''
Created on Apr 8, 2011

@author: duckworthd
'''
import util.classes
import itertools
import numpy as np
    
class GWLocationFF(util.classes.FeatureFunction):
    '''A feature function with a location indicator and
    action indicators'''
    def __init__(self, model):
        # find state with highest average reward
        s_rewards = util.classes.NumMap()
        for (s,a) in itertools.product( model.S(), model.A() ):
            s_rewards[s] += model.R(s,a)/len(model.A())
        self._best_state = s_rewards.argmax()
        
        # action index
        self._index = {}
        for (i,a) in enumerate(model.A()):
            self._index[a] = i+1
        
#        # enumerate all states and all actions
#        S = list( model.S() )
#        A = list( model.A() )
#        self._index = util.classes.NumMap()
#        for (i,s) in enumerate( S ):
#            self._index[s] =  i
#        for (i,a) in enumerate( A ):
#            self._index[a] = len(S)+i
    
    @property
    def dim(self):
        '''dimension of all output from self.features()'''
#        return len(self._index)
        return 1 + len(self._index)
    
    def features(self, s, a):
        '''Calculate features for arguments.  Returns
        numpy.ndarray of length self.dim'''
        result = np.zeros( self.dim )
#        result[ self._index[s] ] = 1
#        result[ self._index[a] ] = 1
#        return result
        s_next = a.apply( s )
        result[0] = np.sum( np.abs( s_next.location - self._best_state.location ) )
#        result[ self._index[a] ] = 1
        return result
        