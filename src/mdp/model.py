from reward import Reward

class State(object):
    '''
    State of an MDP
    '''
    pass

class Action(object):
    '''
    Action in an MDP
    '''
    pass

class Model(object):
    """
    A MDP Model (S,A,T,R,gamma)
    """
    def __init__(self):
        self._gamma = 1.0 
        self._reward_function = Reward()
    
    def T(self,state,action):
        """Returns a function state -> [0,1] for probability of next state
        given currently in state performing action"""
        raise NotImplementedError()
        
    def R(self,state, action):
        """Returns a reward for performing action in state"""
        return self.reward_function.reward(state,action)
        
    def S(self):
        """All states in the MDP"""
        raise NotImplementedError()
        
    def A(self,state=None):
        """All actions in the MDP is state=None, otherwise actions available
        from state"""
        raise NotImplementedError()
        
    @property
    def gamma(self):
        """Discount factor over time"""
        return self._gamma
    
    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma
    
    @property
    def reward_function(self):
        return self._reward_function
    
    @reward_function.setter
    def reward_function(self,rf):
        self._reward_function = rf
    
    def is_terminal(self, state):
        '''returns whether or not a state is terminal'''
        raise NotImplementedError()