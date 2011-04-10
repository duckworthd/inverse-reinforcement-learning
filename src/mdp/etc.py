'''
Created on Apr 9, 2011

@author: duckworthd
'''
import numpy
import util.classes
import mdp.agent

class CompleteFeatureFunction(util.classes.FeatureFunction):
    '''A feature function with indicators for every (s,a) pair'''
    def __init__(self,mdp):
        self._ind = util.classes.NumMap()
        for (i,s) in enumerate( mdp.S() ):
            for (j,a) in enumerate( mdp.A() ):
                self._ind[ (s,a) ] = i*len( mdp.A() ) + j
    @property
    def dim(self):
        '''dimension of all output from self.features()'''
        return len(self._ind)
    
    def features(self, s, a):
        '''Calculate features for arguments.  Returns
        numpy.ndarray of length self.dim'''
        result = numpy.zeros( self.dim )
        result[ self._ind[ (s,a) ] ] = 1.0
        return result
    
class LinearQValueAgent(mdp.agent.Agent):
    '''Implicitly encodes a policy using an MDP's feature function
    and available actions for a given state along with its own weights'''
    def __init__(self, weights, feature_f, actions):
        self._w = weights
        self._ff = feature_f
        self._A = actions
    
    def actions(self,state):
        actions = util.classes.NumMap()
        for a in self._A:
            phi = self._ff.features(state, a)
            actions[a] = numpy.dot( phi, self._w )
        result = util.classes.NumMap()
        result[actions.argmax()] = 1.0
        return result