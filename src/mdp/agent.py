from util.misc import sample
from util.NumMap import NumMap
import sys

class Agent(object):
    """Representation of a policy"""
    
    def actions(self, state):
        """Returns a function from actions -> [0,1] probability of
        performing action in state"""
        raise NotImplementedError()
    
    def sample(self,state):
        """Returns a sample from actions(self,state)"""
        return sample(self.actions(state))
    
class HumanAgent(Agent):
    
    def __init__(self, model):
        self._model = model
    
    def actions(self,state):
        result = NumMap()
        for action in self._model.A(state).keys():
            result[action] = 1.0
        return result.normalize()

    
    def sample(self,state):
        print(state)
        actions = list(self._model.A(state))
        for (i, action) in enumerate(actions):
            print '[%d] %s' % (i, action)
        while True:
            try:
                i = int( sys.stdin.readline() )
                return actions[i]
            except Exception:
                continue
            
class MapAgent(Agent):
    '''
    Agent that always follows a deterministic policy given by a dict
    '''
    
    def __init(self, _policy):
        self._policy = _policy
    
    def actions(self, state):
        """Returns a function from actions -> [0,1] probability of
        performing action in state"""
        result = NumMap()
        result[ self._policy[state] ] = 1.0
        return result