'''
Created on Mar 31, 2011

@author: duckworthd
'''
import util.functions

def simulate(model, agent, initial, t_max):
    '''
    Simulate an MDP for t_max timesteps or until the
    a terminal state is reached.  Returns a list
        [ (s_0, a_0, r_0), (s_1, a_1, r_1), ...]
    '''
    
    s = util.functions.sample(initial)
    result = []
    t = 0
    while t < t_max and not model.is_terminal(s): 
        #TODO Doesn't append terminal state
        a = agent.sample(s)
        s_p = util.functions.sample( model.T(s,a) )
        r = model.R(s,a)
        
        result.append( (s,a,r) )
        s = s_p
        t += 1
    return result


def sample_model(model, n_samples, distr, agent):
    '''
    sample states (s,a,r,s') where s sampled from distribution
    returns
        [(s_0,a_0,r_0,s_p_0), (s_1,a_1,r_1,s_p_1),...]
    '''
    result = []
    for i in range(n_samples):
        s = util.functions.sample(distr)
        a = agent.sample(s)
        r = model.R(s,a)
        s_p = util.functions.sample(model.T(s,a))
        result.append( (s,a,r,s_p) )
    return result