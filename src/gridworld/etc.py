'''
Created on Apr 8, 2011

@author: duckworthd
'''
import util.classes
import numpy
    
class GWLocationFF(util.classes.FeatureFunction):
    '''A feature function with a location indicator and
    action indicators'''
    def __init__(self, mdp):
        # enumerate all states and all actions
        S = list( mdp.S() )
        A = list( mdp.A() )
        self._index = util.classes.NumMap()
        for (i,s) in enumerate( S ):
            self._index[s] =  i
        for (i,a) in enumerate( A ):
            self._index[a] = len(S)+i
    
    @property
    def dim(self):
        '''dimension of all output from self.features()'''
        return len(self._index)
    
    def features(self, s, a):
        '''Calculate features for arguments.  Returns
        numpy.ndarray of length self.dim'''
        result = numpy.zeros( self.dim )
        result[ self._index[s] ] = 1
        result[ self._index[a] ] = 1
        return result