'''
Created on Mar 31, 2011

@author: duckworthd
'''
from util.misc import sample

def simulate(model, agent, initial, t_max):
    '''
    Simulate an MDP for t_max timesteps or until the
    a terminal state is reached.  Returns a list
        [ (s_0, a_0, r_0), (s_1, a_1, r_1), ...]
    '''
    
    s = sample(initial)
    result = []
    t = 0
    while t < t_max and not model.is_terminal(s): 
        #TODO Doesn't append terminal state
        a = agent.sample(s)
        s_p = sample( model.T(s,a) )
        r = model.R(s,a)
        
        result.append( (s,a,r) )
        s = s_p
        t += 1
    return result