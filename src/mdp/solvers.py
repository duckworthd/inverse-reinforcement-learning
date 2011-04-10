'''
Created on Apr 7, 2011

@author: duckworthd
'''
from util.classes import NumMap
from mdp.agent import Agent, MapAgent, RandomAgent
from numpy import dot, outer, zeros, eye, vstack
import numpy.random
from numpy.linalg import pinv
import random
from mdp import simulation

class ExactSolver(object):
    def solve(self, mdp):
        '''Returns an Agent directed by a policy determined by this solver'''
        raise NotImplementedError()
    
class ApproximateSolver(object):
    def solve(self, mdp):
        '''Approximately solve an MDP via samples generated by simulation.
        a linear reward function is assume, and reward_f is thus assumed to
        be of class LinearReward.  Only its feature function will be used.'''
        raise NotImplementedError()   

class IRLExactSolver(object):
    '''Solves the inverse reinforcement learning problem'''
    def solve(self, mdp, feature_f, agent):
        '''
        Returns a pair (agent, weights) where the agent attempts to mimic the
        argument agent via solving an MDP with R(s,a) = feature_f.features(s,a)*weights
        
        feature_f: linear feature function for which weights will be learned
        agent: optimal agent for which we try to learn rewards for
        '''
    
class IRLApprximateSolver(object):
    '''Solves the inverse reinforcement learning problem'''
    def solve(self, mdp, feature_f, samples):
        '''
        Returns a pair (agent, weights) where the agent attempts to generalize
        from the behavior observed in samples and weights is what was combined with
        the MDP to generate agent.
        
        feature_f: linear feature function for which weights will be learned
        samples: sample trajectories [ (s_t,a_t) ] of the (supposedly) optimal
            policy.
        '''
    
class ValueIterator(ExactSolver):
    def __init__(self, max_iter):
        self._max_iter = max_iter
        
    def solve(self, mdp):
        V = NumMap()
        for i in range(self._max_iter):
            V = self.iter(mdp, V, 'max')
        return MapAgent(self.iter(mdp, V, 'argmax'))
        
    @classmethod
    def iter(cls, mdp, V, max_or_argmax='max'):
        ''' 1 step lookahead via the Bellman Update.  final argument should
        be either 'max' or 'argmax', determining whether a state-value function
        or a policy is returned'''
        if max_or_argmax == 'max':
            V_next = NumMap()
        else:
            V_next = {}
        for s in mdp.S():
            if mdp.is_terminal(s):
                V[s] = 0.0
                continue
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

class QValueIterator(ExactSolver):
    '''Use Q-Value Iteration to solve an MDP'''
    def __init__(self, max_iter):
        self._max_iter = max_iter
        
    def solve(self, mdp):
        '''Returns an Agent directed by a policy determined by this solver'''
        Q = NumMap()
        for i in range(self._max_iter):
            Q = self.iter(mdp, Q)
        policy = {}
        for s in mdp.S():
            actions = NumMap()
            for a in mdp.A(s):
                actions[a] = Q[ (s,a) ]
            policy[s] = actions.argmax()
        return MapAgent(policy)
    
    @classmethod
    def iter(cls, mdp, Q):
        V = NumMap()
        # Compute V(s) = max_{a} Q(s,a)
        for s in mdp.S():
            V_s = NumMap()
            for a in mdp.A(s):
                V_s[a] = Q[ (s,a) ]
            if len(V_s) > 0:
                V[s] = V_s.max()
            else:
                V[s] = 0.0
        
        # QQ(s,a) = R(s,a) + gamma*sum_{s'} T(s,a,s')*V(s') 
        QQ = NumMap()
        for s in mdp.S():
            for a in mdp.A(s):
                value = mdp.R(s,a)
                T = mdp.T(s,a)
                value += sum( [mdp.gamma*t*V[s_prime] for (s_prime,t) in  T.items()] )
                QQ[ (s,a) ] = value
        return QQ
    
class LinearQValueAgent(Agent):
    '''Implicitly encodes a policy using an MDP's feature function
    and available actions for a given state along with its own weights'''

    def __init__(self, weights, feature_f, actions):
        self._w = weights
        self._ff = feature_f
        self._A = actions
    
    def actions(self,state):
        actions = NumMap()
        for a in self._A:
            phi = self._ff.features(state, a)
            actions[a] = dot( phi, self._w )
        result = NumMap()
        result[actions.argmax()] = 1.0
        return result
        
class LSPI(ApproximateSolver):
    '''Least Squares Policy Iteration (Lagoudakis, Parr 2001)'''
    
    def __init__(self, n_iter, n_samples):
        self._n_iter = n_iter
        self._n_samples = n_samples
        
    def solve(self, mdp, feature_f):
        '''Also requires a feature function describing the state state space'''
        # initialize policy randomly
        k = feature_f.dim
        w = numpy.random.rand( k )
        agent = LinearQValueAgent(w, feature_f, mdp.A())

        # initial state distribution
        s = random.choice(mdp.S())
        initial = NumMap( {s:1.0} )
        
        # Generate samples
        samples = simulation.simulate(mdp, RandomAgent( mdp.A() ), initial, self._n_samples)
        ## Does the a in (s,a,r,s') need to follow pi(s)?  
        ##     NO! lspi-short.pdf, 3/4 down page 4
        ## If deterministic policy used, what about missing (s,a) 
        ## for a != pi(s), equivalent to zero in stationary distribution 
        ## of (s,a)~MDP|pi?
        
        for i in range(self._n_iter):
            # evaluate policy approximately
            w = self.lstdq(samples, feature_f, mdp.gamma, agent)
            ## Is this correct?  implement exactly first
#            w = self.lstdq_exact(mdp, agent, feature_f)
            
            # Define an agent to use argmax over Q(s,a) to choose actions
            agent = LinearQValueAgent(w,feature_f, mdp.A())
             
        return LinearQValueAgent(w,feature_f, mdp.A())
    
    def lstdq(self, samples, feature_f, gamma, agent):
        '''find weights to approximate value function of a given policy
        Q(s,a) ~~ dot(w,phi(s,a))'''
        k = feature_f.dim
        A = zeros( [k,k] )
        b = zeros( k )
        for i in range(len(samples)-1):
            (s,a,r) = samples[i]
            s_prime = samples[i+1][0]
            phi = feature_f.features(s,a)
            phi_prime = feature_f.features(s_prime, agent.sample(s_prime))
            A = A + outer(phi,phi - gamma*phi_prime)
            b = b + phi*r
        w = dot( pinv(A), b)
        return w
    
    def lstdq_exact(self, mdp, agent, feature_f):
        '''exact version of lstdq for sanity checking.  Agent assumed deterministic'''
        S = list( mdp.S() )
        A = list( mdp.A() )
        
        k = feature_f.dim
        PHI = zeros( [len(S)*len(A), k] )       # each row is phi(s,a)
        P_PHI = zeros( [len(S)*len(A), k] )     # each row is sum_{s'} P(s'|s,a)*PHI(s',pi(s'))
        R = zeros( [len(S)*len(A), 1] )         # each row is R(s,a)
        for (i,s) in enumerate(S):
            for (j,a) in enumerate(A):
                PHI[i*len(A)+j,:] = feature_f.features(s,a).T
                for (s_p,t) in mdp.T(s,a).items():
                    P_PHI[i*len(A)+j,:] += t*feature_f.features(s_p,agent.sample(s_p)).T 
                R[i*len(A)+j,0] = mdp.R(s,a)
        A = dot( PHI.T, PHI - mdp.gamma*P_PHI )
        b = dot( PHI.T, R )
        w = dot( pinv(A), b)    # should be weights least squares according to stationary distr.
        return w
        
        