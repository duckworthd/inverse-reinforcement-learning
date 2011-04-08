'''
Created on Apr 7, 2011

@author: duckworthd
'''
from util.NumMap import NumMap
from mdp.agent import MapAgent

class Solver(object):
    def solve(self, mdp):
        '''Returns an Agent directed by a policy determined by this solver'''
        raise NotImplementedError()
    
    
class ValueIterator(Solver):
    def __init__(self, max_iter):
        self._max_iter = max_iter
        
    def solve(self, mdp):
        V = NumMap()
        for i in range(self._max_iter):
            V = self.iter(mdp, V, 'max')
        return MapAgent(self.iter(mdp, V, 'argmax'))
        
    def iter(self, mdp, V, max_or_argmax='max'):
        ''' 1 step lookahead via the Bellman Update.  final argument should
        be either 'max' or 'argmax', determining whether a state-value function
        or a policy is returned'''
        if max_or_argmax == 'max':
            V_next = NumMap()
        else:
            V_next = {}
        for s in mdp.S():
            q = NumMap()    # Q-states for state s
            for a in mdp.A(s):
                r = mdp.R(s,a)
                T = mdp.T(s,a)
                expected_rewards = [T[s_prime]*V[s_prime] for s_prime in T]
                q[a] = r + mdp.gamma*sum(expected_rewards)
            if max_or_argmax == 'max':
                V_next[s] = q.max()
            else:
                V_next[s] = q.argmax()
        return V_next
                    