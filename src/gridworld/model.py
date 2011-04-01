from mdp.model import *
from numpy import array, all, zeros, logical_and
from util.NumMap import NumMap
import itertools

class GWState(State):
    
    def __init__(self, location=(0,0)):
        self._location = location
    
    @property
    def location(self):
        return self._location
    
    @location.setter
    def location(self, loc):
        self._location = loc
        
    def __str__(self):
        return 'GWState: [location={}]'.format(self.location)
    
    def __repr(self):
        return self.__str__()
    
    def __eq__(self, other):
        try:
            return all(self.location == other.location)
        except Exception:
            return False
    
    def __hash__(self):
        return self.location.__hash__()
    

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
    
    def __repr(self):
        return self.__str__()
    
    def __eq__(self,other):
        try:
            return all(self.direction == other.direction)
        except Exception:
            return False
    
    def __hash__(self):
        return self.direction().__hash__()
        
        
class GWModel(Model):
    
    def __init__(self, p_fail=0.2, map_size=(4,3), terminal=(-1,-1)):
        super(GWModel,self)
        
        up      = array( [0, 1] )
        down    = array( [0,-1] )
        left    = array( [1, 0] )
        right   = array( [-1,0] ) 
        
        self._actions = [GWAction(up), GWAction(down), GWAction(left), GWAction(right)]
        self._p_fail = float(p_fail)
        self._map_size = map_size
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
        
        
    def R(self,state, action):
        """Returns a reward for performing action in state"""
        return self.reward_function.reward(state,action)
        
    def S(self):
        """All states in the MDP"""
        return [GWState(loc) for loc in itertools.product(range(self._map_size[0]), range(self._map_size[1]) )]
        
    def A(self,state=None):
        """All actions in the MDP is state=None, otherwise actions available
        from state"""
        return self._actions
    
    def is_terminal(self, state):
        '''returns whether or not a state is terminal'''
        return all(state.location == self._terminal)
    
    def is_legal(self,state):
        return all( logical_and(state.location >= zeros(2), state.location < self._map_size ) )
    
    def __str__(self):
        format = 'GWModel [p_fail={},map_size={},terminal={}]'
        return format.format(self._p_fail,self._map_size,self._terminal)