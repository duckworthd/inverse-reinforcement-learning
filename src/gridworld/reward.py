from mdp.reward import LinearReward
import numpy
from model import GWState, GWAction

class GWBoxReward(LinearReward):
    '''
    Feature functions are whether the agent is in
    a box of a specific size
    '''
    
    def __init__(self, box_size, map_size):
        super(GWBoxReward,self).__init__()
        self._box_size = numpy.array(box_size, dtype='float32')
        self._map_size = map_size
    
    def dim(self):
        return  int( numpy.prod( numpy.ceil( self._map_size/self._box_size ) ) )
    
    def features(self, state, action):
        # How many boxes in a single row
        box_per_row = self._map_size[1]/self._box_size[1]
        
        # what's location of agent in box coordinates
        box_loc = numpy.floor(state.location/self._box_size)
        
        result = numpy.zeros( self.dim() )
        result[box_loc[0] + box_loc[1]*box_per_row ] = 1
        return result
    
        
    def __str__(self):
        result = 'GWBoxReward:\n'
        for i in reversed(range(self._map_size[0])):
            for j in range(self._map_size[1]):
                state = GWState( numpy.array( [i,j] ) )
                action = GWAction( numpy.array([0,0]) )
                result += '|{: 4.4f}|'.format(self.reward(state, action))
            result += '\n'
        return result
                