from util.functions import sample
from util.classes import NumMap
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
        raise NotImplementedError()
    
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
    
    def __init__(self, _policy):
        self._policy = _policy
    
    def actions(self, state):
        """Returns a function from actions -> [0,1] probability of
        performing action in state"""
        result = NumMap()
        result[ self._policy[state] ] = 1.0
        return result
    
class QValueAgent(Agent):
    '''Agent that uses a policy implicitly encoded in a Q-function'''

    def __init__(self, Q):
        '''
        Q: a map (state,action) -> [0,1]
        '''
        # build a two-stage dictionary for quick processing
        states = set([s for (s,a) in Q])
        actions = set([a for (s,a) in Q])
        
        QQ = {}
        for s in states:
            QQ[s] = NumMap()
        for s in states:
            for a in actions:
                if (s,a) in Q:
                    QQ[s][a] = Q[(s,a)]
        self._Q = QQ
    
    def actions(self, state):
        """Returns a function from actions -> [0,1] probability of
        performing action in state"""
        result = NumMap()
        result[ self._Q[state].argmax() ] = 1.0
        return result

class RandomAgent(Agent):
    
    def __init__(self, actions):
        self._action_distr = NumMap()
        for a in actions:
            self._action_distr[a] = 1.0
        self._action_distr = self._action_distr.normalize()
    
    def actions(self, state):
        """Returns a function from actions -> [0,1] probability of
        performing action in state"""
        return self._action_distr