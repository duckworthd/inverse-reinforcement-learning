from mdp.reward import LinearReward
import numpy
from model import GWState, GWAction, GWModel

class GWBoxReward(LinearReward):
    '''
    Feature functions are whether the agent is in
    a box of a specific size
    '''
    
    def __init__(self, box_size, map):
        super(GWBoxReward,self).__init__()
        self._box_size = numpy.array(box_size, dtype='float32')
        self._map_size = numpy.array( map.shape )
        self._actions = list( GWModel(0.0).A() )
    
    @property
    def dim(self):
        return  int( numpy.prod( numpy.ceil( self._map_size/self._box_size ) )+len(self._actions) )
    
    def features(self, state, action):
        # How many boxes in a single row
        box_per_row = self._map_size[1]/self._box_size[1]
        
        # what's location of agent in box coordinates
        box_loc = numpy.floor(state.location/self._box_size)
        
        # location indicator
        result = numpy.zeros( self.dim )
        result[box_loc[0]*box_per_row + box_loc[1] ] = 1
        
        # action indicator
        action_i = [i for (i,a) in enumerate(self._actions) if a == action][0]
        result[len(result)-len(self._actions)+action_i] = 1
        
        return result
    
    def __str__(self):
        return 'GWBoxReward [box_size={}, map_size={}]'.format(self._box_size, self._map_size)
        
    def info(self):
        result = 'GWBoxReward:\n'
        for a in self._actions:
            result += str(a) + '\n'
            for i in reversed(range(self._map_size[0])):
                for j in range(self._map_size[1]):
                    state = GWState( numpy.array( [i,j] ) )
                    action = a
                    result += '|{: 4.4f}|'.format(self.reward(state, action))
                result += '\n'
            result += '\n\n'
        return result
                