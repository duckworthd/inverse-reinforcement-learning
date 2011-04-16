'''
Created on Apr 7, 2011

@author: duckworthd
'''
import util.classes
import mdp.agent
import numpy as np
import numpy.random
import numpy.linalg
import random
import mdp.simulation
import mdp.etc
import math
import itertools

class MDPSolver(object):
    def solve(self, model):
        '''Returns an Agent that maximizes the expected reward over time, ie
        max_{pi} E[sum_t gamma^t r_t | pi]''' 
        raise NotImplementedError() 

class PolicyEvaluator(object):
    def evaluate_policy(self, model, agent):
        '''Evaluate a policy and return a function s --> R such
        V(s) = E[sum_t gamma^t r_t | pi]'''
        raise NotImplementedError()

class IRLExactSolver(object):
    '''Solves the inverse reinforcement learning problem'''
    def __init__(self, max_iter, mdp_solver):
        self._max_iter = max_iter
        self._solver = mdp_solver
    
    def solve(self, model, initial, true_agent):
        '''
        Returns a pair (agent, weights) where the agent attempts to mimic the
        argument agent via solving an MDP with R(s,a) = feature_f.features(s,a)*weights
        
        mdp: an MDP with a linear reward functions.  Parameters WILL be overwritten.
        initial: initial distribution over states
        agent: optimal agent for which we try to learn rewards for
        '''
        # Compute feature expectations of agent = mu_E from samples
        mu_E = self.feature_expectations(model, initial, true_agent)
        
        # Pick random policy pi^(0)
        agent = mdp.agent.RandomAgent( model.A() )
        
        # Calculate feature expectations of pi^(0) = mu^(0)
        mu = self.feature_expectations(model, initial, agent)
        
        for i in range(self._max_iter):
            # TODO: Use CVXOPT for max-margin method
            
            # Perform projections to new weights w^(i)
            if i == 0:
                mu_bar = mu
            else:
                mmmb = mu - mu_bar
                mu_bar = mu_bar + numpy.dot( mmmb, mu_E-mu_bar )/numpy.dot( mmmb,mmmb )*mmmb
            w = mu_E - mu_bar
            t = numpy.linalg.norm(mu_E - mu_bar)
            
            print 't = %f' % (t,)
            if t < 1e-10:
                break
            
            # Compute optimal policy used R(s,a) = dot( feature_f(s,a), w^(i) )
            model.reward_function.params = w
            agent = self._solver.solve(model)
            
            # Compute feature expectations of pi^(i) = mu^(i)
            mu = self.feature_expectations(model, initial, agent)
        
        return (agent, w)
    
    def feature_expectations(self, model, initial, agent):
        ff = model.reward_function
        i = 0
        # Initialize feature expectations
        mu = {}
        for s in model.S():
            for a in model.A(s):
                mu[ (s,a) ] = numpy.zeros( ff.dim )
        
        # Until error is less than 1% (assuming ||phi(s,a)||_{inf} <= 1 for all (s,a) )
        # mu(s,a) = phi(s,a) + gamma*sum_{s'} P(s'|s,a) *sum_{a'} P(a'|s') mu(s',a')
        while model.gamma**i >= 0.01:
            i += 1
            mu2 = {}
            for s in model.S():
                for a in model.A(s):
                    v = ff.features(s,a)
                    for (s_prime,t_s) in model.T(s,a).items():
                        for (a_prime,t_a) in agent.actions(s_prime).items():
                            v += model.gamma*t_s*t_a*mu[ (s_prime, a_prime) ]
                    mu2[ (s,a) ] = v
            mu = mu2
        result = numpy.zeros( ff.dim )
        # result = sum_{s} P(s) * sum_{a} P(a|s)*mu(s,a)
        for s in initial:
            pi = agent.actions(s)
            for a in pi:
                result += initial[s]*pi[a]*mu[ (s,a) ]
        return result
            
class IRLApprximateSolver(object):
    '''Solves the inverse reinforcement learning problem'''
    def __init__(self, max_iter, mdp_solver, n_samples=500):
        '''
        max_iter: maximum number of times to iterate policies
        mdp_solver: class that implements self.solve(model)
        n_samples: number of samples used to estimate feature expectations
        '''
        self._max_iter = max_iter
        self._solver = mdp_solver
        self._n_samples = n_samples
        
    def solve(self, model, initial, true_samples):
        '''
        Returns a pair (agent, weights) where the agent attempts to generalize
        from the behavior observed in samples and weights is what was combined with
        the MDP to generate agent.
        
        mdp: an MDP with a linear reward functions.  Parameters WILL be overwritten.
        initial: initial distribution over states
        samples: sample trajectories [ (s_t,a_t) ] of the (supposedly) optimal
            policy.
        '''
        # Compute feature expectations of agent = mu_E from samples
        mu_E = self.feature_expectations(model, true_samples)
        
        # Pick random policy pi^(0)
        agent = mdp.agent.RandomAgent( model.A() )
        
        # Calculate feature expectations of pi^(0) = mu^(0)
        samples = self.generate_samples(model, agent, initial)
        mu = self.feature_expectations(model, samples)
        
        for i in range(self._max_iter):
            # Perform projections to new weights w^(i)
            if i == 0:
                mu_bar = mu
            else:
                mmmb = mu - mu_bar
                mu_bar = mu_bar + numpy.dot( mmmb, mu_E-mu_bar )/numpy.dot( mmmb,mmmb )*mmmb
            w = mu_E - mu_bar
            t = numpy.linalg.norm(mu_E - mu_bar)
            model.reward_function.params = w
            
            print 't = %f' % (t,)
            
            # Compute optimal policy used R(s,a) = dot( feature_f(s,a), w^(i) )
            agent = self._solver.solve(model)
            
            # Compute feature expectations of pi^(i) = mu^(i)
            samples = self.generate_samples(model, agent, initial)
            mu = self.feature_expectations(model, samples)
        
        return (agent, w)
    
    def generate_samples(self, model, agent, initial):
        # t_max such that gamma^t_max = 0.01
        t_max = math.ceil( math.log(0.01)/math.log(model.gamma) )
        result = []
        for i in range(self._n_samples):
            result.append( mdp.simulation.simulate(model, agent, initial, t_max) )
        return result
            
    def feature_expectations(self, model, samples):
        '''Compute empirical feature expectations'''
        ff = model.reward_function
        result = numpy.zeros( ff.dim )
        for sample in samples:
            for (t,sa) in enumerate(sample):
                s = sa[0]
                a = sa[1]
                result += (model.gamma**t)*ff.features(s,a)
        return (1.0/len(samples))*result
    
class ValueIterator(MDPSolver):
    def __init__(self, max_iter):
        self._max_iter = max_iter
        
    def solve(self, model):
        V = util.classes.NumMap()
        for i in range(self._max_iter):
            V = self.iter(model, V, 'max')
            print 'Iteration #{}'.format(i)
        return mdp.agent.MapAgent(self.iter(model, V, 'argmax'))
        
    @classmethod
    def iter(cls, model, V, max_or_argmax='max'):
        ''' 1 step lookahead via the Bellman Update.  final argument should
        be either 'max' or 'argmax', determining whether a state-value function
        or a policy is returned'''
        if max_or_argmax == 'max':
            V_next = util.classes.NumMap()
        else:
            V_next = {}
        for s in model.S():
            if model.is_terminal(s):
                V[s] = 0.0
                continue
            q = util.classes.NumMap()    # Q-states for state s
            for a in model.A(s):
                r = model.R(s,a)
                T = model.T(s,a)
                expected_rewards = [T[s_prime]*V[s_prime] for s_prime in T]
                q[a] = r + model.gamma*sum(expected_rewards)
            if max_or_argmax == 'max':
                V_next[s] = q.max()
            else:
                V_next[s] = q.argmax()
        return V_next

class QValueIterator(MDPSolver):
    '''Use Q-Value Iteration to solve an MDP'''
    def __init__(self, max_iter):
        self._max_iter = max_iter
        
    def solve(self, model):
        '''Returns an Agent directed by a policy determined by this solver'''
        Q = util.classes.NumMap()
        for i in range(self._max_iter):
            Q = self.iter(model, Q)
        policy = {}
        for s in model.S():
            actions = util.classes.NumMap()
            for a in model.A(s):
                actions[a] = Q[ (s,a) ]
            policy[s] = actions.argmax()
        return mdp.agent.MapAgent(policy)
    
    @classmethod
    def iter(cls, model, Q):
        V = util.classes.NumMap()
        # Compute V(s) = max_{a} Q(s,a)
        for s in model.S():
            V_s = util.classes.NumMap()
            for a in model.A(s):
                V_s[a] = Q[ (s,a) ]
            if len(V_s) > 0:
                V[s] = V_s.max()
            else:
                V[s] = 0.0
        
        # QQ(s,a) = R(s,a) + gamma*sum_{s'} T(s,a,s')*V(s') 
        QQ = util.classes.NumMap()
        for s in model.S():
            for a in model.A(s):
                value = model.R(s,a)
                T = model.T(s,a)
                value += sum( [model.gamma*t*V[s_prime] for (s_prime,t) in  T.items()] )
                QQ[ (s,a) ] = value
        return QQ

class IteratingPolicyEvaluator(PolicyEvaluator):
    def __init__(self, max_iter):
        self._max_iter = max_iter
          
    def evaluate_policy(self, model, agent):
        '''Evaluate a policy and return a function s --> R such
        V(s) = E[sum_t gamma^t r_t | pi]'''
        V = util.classes.NumMap()
        for i in range(self._max_iter):
            V = self.iter(model, agent, V)
        return V
    
    def iter(self, model, agent, V):
        VV = util.classes.NumMap()
        for s in model.S():
            pi = agent.actions(s)
            vv = 0
            for (a,t_pi) in pi.items():
                v = model.R(s,a)
                T = model.T(s,a)
                v += model.gamma*sum( [t*V[s_prime] for (s_prime,t) in T.items()] )
                vv += t_pi*v
            VV[s] = vv
        return VV

class ExactPolicyEvaluator(PolicyEvaluator):
    def evaluate_policy(self, model, agent):
        '''
        Use linear algebra to solve
            Q = R + gamma*P*P*Q
        where R  is (m*n x 1)
              P  is (m*n x n)
              PI is (n x m*n) with n copies of P(a=i|s=j) along each row
              Q  is (m*n x 1)
        m = number of actions
        n = number of states
        '''
        # State + Actions
        S = list( model.S() )
        A = list( model.A() )
        SA = []
        for s in S:
            for a in A:
                SA.append((s,a))
        
        S_dict = {}
        for (i,s) in enumerate(S):
            S_dict[s] = i
        
        A_dict = {} 
        for (j,a) in enumerate(A):
            A_dict[a] = j
        
        SA_dict = {}
        for (i,s) in enumerate(S):
            for (j,a) in enumerate(A):
                SA_dict[(s,a)] = i*len(A)+j
        
        gamma = model.gamma
        (n,m) = ( len(S), len(A) )
        R = np.zeros( m*n )
        P = np.zeros( [m*n,n])
        PI= np.zeros( [n,m*n])        
        
        # Fill R
        for ((s,a),i) in SA_dict.items():
            R[i] = model.R(s,a)
        
        # Fill P
        for ((s,a),i) in SA_dict.items():
            T = model.T(s,a)
            for (s2,p) in T.items():
                j = S_dict[s2]
                P[i,j] = p
        
        # Fill PI
        pis = {}
        for s in S:
            pi = agent.actions(s)
            for a in A:
                pis[(s,a)] = pi[a]
        for (i,s) in enumerate(S):
            for (j,(s_p,a)) in enumerate(SA):
                if s_p != s:
                    continue
                PI[i,j] = pis[(s,a)]
        
        # Solve
        Q = numpy.linalg.solve(np.eye(n*m)-gamma*np.dot(P,PI), R) 
        
        # Build V = max_{A} Q(s,a)
        V = util.classes.NumMap()
        for (i,s) in enumerate(S):
            acts = util.classes.NumMap()
            for a in A:
                acts[a] = Q[ SA_dict[(s,a)] ]
            V[s] = acts.max()
        return V
                
                
class SamplingPolicyEvaluator(PolicyEvaluator):
    def __init__(self, n_samples, sample_len=-1):
        self._n_samples = n_samples
        self._sample_len = sample_len
        
    def evaluate_policy(self, model, agent):
        '''Evaluate a policy and return a function s --> R such
        V(s) = E[sum_t gamma^t r_t | s_0=s, pi]'''
        if self._sample_len < 0:
            sample_len = math.log(0.01)/math.log(model.gamma)
        else:
            sample_len = self._sample_len
            
        V = util.classes.NumMap()
        for s in model.S():
            v = 0
            # Simulate n experience starting from this state, following pi
            initial = util.classes.NumMap( {s:1.0} )
            for i in range(self._n_samples):
                sample = mdp.simulation.simulate(model, agent, initial, sample_len)
                for (t, (s,a,r)) in enumerate( sample ):
                    v += (model.gamma**t)*r
            # V(s) = (1/m)* sum_{i=1}^{t} sum_t r_t*gamma^t
            #     where m = number of simulations
            V[s] = v / self._n_samples
        return V
        
            
class PolicyIterator(MDPSolver):
    '''Performs exact policy iterations.  Policies are lookup tables.'''
    def __init__(self, max_iter, policy_eval):
        self._max_iter = max_iter
        self._eval = policy_eval
        
    def solve(self, model):
        '''Returns an Agent directed by a policy determined by this solver'''
        agent = mdp.agent.RandomAgent(model.A())
        for i in range(self._max_iter):
            # Evaluate policy
            V_pi = self._eval.evaluate_policy(model, agent)
            
            print 'Iteration #{}'.format(i)
            
            # Do a 1-step lookahead, choose best action
            policy = {}
            for s in model.S():
                actions = util.classes.NumMap()
                for a in model.A(s):
                    v = model.R(s,a)
                    T = model.T(s,a)
                    v += sum( [model.gamma*t*V_pi[s_prime] for (s_prime,t) in T.items()] )
                    actions[a] = v
                policy[s] = actions.argmax()
            agent = mdp.agent.MapAgent(policy)
        return agent
        
class LSPI(MDPSolver):
    '''Least Squares Policy Iteration (Lagoudakis, Parr 2001)'''
    
    def __init__(self, n_iter, n_samples, feature_f=None):
        self._n_iter = n_iter
        self._n_samples = n_samples
        self._feature_f = feature_f
        
    def solve(self, model):
        '''Also requires a feature function describing the state state space'''
        if self._feature_f == None:
            feature_f = mdp.etc.CompleteFeatureFunction(model)
        else:
            feature_f = self._feature_f
        
        # initialize policy randomly
        k = feature_f.dim
        w = numpy.random.rand( k )
        agent = mdp.etc.LinearQValueAgent(w, feature_f, model.A())

        # initial state distribution
        s = random.choice(model.S())
        initial = util.classes.NumMap( {s:1.0} )
        
        # Generate samples
        samples = mdp.simulation.simulate(model, mdp.agent.RandomAgent( model.A() ), initial, self._n_samples)
        ## Does the a in (s,a,r,s') need to follow pi(s)?  
        ##     NO! lspi-short.pdf, 3/4 down page 4
        ## If deterministic policy used, what about missing (s,a) 
        ## for a != pi(s), equivalent to zero in stationary distribution 
        ## of (s,a)~MDP|pi?
        
        for i in range(self._n_iter):
            # evaluate policy approximately
            w = self.lstdq(samples, feature_f, model.gamma, agent)
            ## Is this correct?  implement exactly first
#            w = self.lstdq_exact(model, agent, feature_f)
            
            # Define an agent to use argmax over Q(s,a) to choose actions
            agent = mdp.etc.LinearQValueAgent(w,feature_f, model.A())
             
        return mdp.etc.LinearQValueAgent(w,feature_f, model.A())
    
    def lstdq(self, samples, feature_f, gamma, agent):
        '''find weights to approximate value function of a given policy
        Q(s,a) ~~ dot(w,phi(s,a))'''
        k = feature_f.dim
        A = numpy.zeros( [k,k] )
        b = numpy.zeros( k )
        for i in range(len(samples)-1):
            (s,a,r) = samples[i]
            s_prime = samples[i+1][0]
            phi = feature_f.features(s,a)
            phi_prime = feature_f.features(s_prime, agent.sample(s_prime))
            A = A + numpy.outer(phi,phi - gamma*phi_prime)
            b = b + phi*r
        w = numpy.dot( numpy.linalg.pinv(A), b)
        return w
    
    def lstdq_exact(self, model, agent, feature_f):
        '''exact version of lstdq for sanity checking.  Agent assumed deterministic'''
        S = list( model.S() )
        A = list( model.A() )
        
        k = feature_f.dim
        PHI = numpy.zeros( [len(S)*len(A), k] )       # each row is phi(s,a)
        P_PHI = numpy.zeros( [len(S)*len(A), k] )     # each row is sum_{s'} P(s'|s,a)*PHI(s',pi(s'))
        R = numpy.zeros( [len(S)*len(A), 1] )         # each row is R(s,a)
        for (i,s) in enumerate(S):
            for (j,a) in enumerate(A):
                PHI[i*len(A)+j,:] = feature_f.features(s,a).T
                for (s_p,t) in model.T(s,a).items():
                    P_PHI[i*len(A)+j,:] += t*feature_f.features(s_p,agent.sample(s_p)).T 
                R[i*len(A)+j,0] = model.R(s,a)
        A = numpy.dot( PHI.T, PHI - model.gamma*P_PHI )
        b = numpy.dot( PHI.T, R )
        w = numpy.dot( numpy.linalg.pinv(A), b)    # should be weights least squares according to stationary distr.
        return w
        
        