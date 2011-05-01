from mdp.model import *
import numpy as np
from numpy import array, all, zeros, logical_and, ones
from util.classes import NumMap
import itertools

class GWState(State):
    
    def __init__(self, location=array( [0,0] )):
        self._location = location
    
    @property
    def location(self):
        return self._location
    
    @location.setter
    def location(self, loc):
        self._location = loc
        
    def __str__(self):
        return 'GWState: [location={}]'.format(self.location)
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        try:
            return all( self.location == other.location) # epsilon error
        except Exception:
            return False
    
    def __hash__(self):
        loc = self.location # hash codes for numpy.array not consistent?
        return (loc[0], loc[1]).__hash__()
    

class GWAction(Action):
    
    def __init__(self, direction):
        self._direction = direction
     
    @property   
    def direction(self):
        return self._direction
    
    @direction.setter
    def direction(self, dir):
        self._direction = dir
        
    def apply(self,gwstate):
        return GWState( gwstate.location + self.direction )
    
    def __str__(self):
        return "GWAction: [direction={}]".format(self.direction)
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self,other):
        try:
            return all(self.direction == other.direction)
        except Exception:
            return False
    
    def __hash__(self):
        dir = self.direction    # hash codes for numpy.array not consistent?
        return (dir[0], dir[1]).__hash__()
        
        
class GWModel(Model):
    
    def __init__(self, p_fail=0.2, map=ones( [4,3] ), 
                 terminal=GWState(np.array( [-1,-1] )) ):
        super(GWModel,self)
        
        up      = array( [0, 1] )
        down    = array( [0,-1] )
        left    = array( [1, 0] )
        right   = array( [-1,0] ) 
        
        self._actions = [GWAction(up), GWAction(down), GWAction(left), GWAction(right)]
        self._p_fail = float(p_fail)
        self._map = map
        self._terminal = terminal
    
    def T(self,state,action):
        """Returns a function state -> [0,1] for probability of next state
        given currently in state performing action"""
        result = NumMap()
        actions = self.A(state)
        for a in actions:
            p = 0
            if a == action:
                p = 1 - self._p_fail
            else:
                p = self._p_fail / ( len(actions)-1 )
            s_p = a.apply(state)
            if not self.is_legal(s_p):
                result[state] += p
            else:
                result[s_p] += p  
        return result 
        
    def S(self):
        """All states in the MDP"""
        result = []
        nz = np.transpose( np.nonzero(self._map == 1) )
        for ind in nz:
            result.append( GWState( ind ) )
        return result
        
    def A(self,state=None):
        """All actions in the MDP is state=None, otherwise actions available
        from state"""
        return self._actions
    
    def is_terminal(self, state):
        '''returns whether or not a state is terminal'''
        return all(state == self._terminal)
    
    def is_legal(self,state):
        loc = state.location
        (r,c) = self._map.shape
        
        return loc[0] >= 0 and loc[0] < r and \
            loc[1] >= 0 and loc[1] < c and \
            self._map[ loc[0],loc[1] ] == 1
    
    def __str__(self):
        format = 'GWModel [p_fail={},terminal={}]'
        return format.format(self._p_fail, self._terminal)
    
    def info(self):
        result = [str(self) + '\n']
        map_size = self._map.shape
        for i in reversed(range(map_size[0])):
            for j in range(map_size[1]):
                if self._map[i,j] == 1:
                    result.append( '[O]' )
                else:
                    result.append( '[X]')
            result.append( '\n' )
        return ''.join(result)
