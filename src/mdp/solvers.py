'''
Created on Apr 7, 2011

@author: duckworthd
'''
import itertools
import math
import mdp.agent
import mdp.etc
import mdp.simulation
import numpy as np
import numpy.linalg
import numpy.random
import random
import util.classes

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
        w_0 = model.reward_function.params
        agents = []
        
        # Compute feature expectations of agent = mu_E from samples
        mu_E = self.feature_expectations2(model, initial, true_agent)
        
        # Pick random policy pi^(0)
        agent = mdp.agent.RandomAgent( model.A() )
        
        # Calculate feature expectations of pi^(0) = mu^(0)
        mu = self.feature_expectations2(model, initial, agent)
        
        print mu_E, mu
        
        for i in range(self._max_iter):
            agents.append(agent)
            # TODO: Use CVXOPT for max-margin method
            
            # Perform projections to new weights w^(i)
            if i == 0:
                mu_bar = mu
            else:
                mmmb = mu - mu_bar
                mu_bar = mu_bar + numpy.dot( mmmb, mu_E-mu_bar )/numpy.dot( mmmb,mmmb )*mmmb
            w = mu_E - mu_bar
            t = numpy.linalg.norm(mu_E - mu_bar)
            
            print 'IRLExactSolver Iteration #{};t = {:4.4f}'.format(i, t)
            if t < 1e-10:
                break
            
            # Compute optimal policy used R(s,a) = dot( feature_f(s,a), w^(i) )
            model.reward_function.params = w
            agent = self._solver.solve(model)
            
            # Compute feature expectations of pi^(i) = mu^(i)
            mu = self.feature_expectations2(model, initial, agent)
        
        # Restore params
        model.reward_function.params = w
        return (agent, w)
    
    def feature_expectations(self, model, initial, agent):
        '''
        Calculate mu(s,a) = E[sum_t gamma^ phi(s_t,a_t) | s_0=s, a_0=a] via repeated applications of
            mu(s,a) <--- phi(s,a) + gamma*sum_s' P(s'|s,a) P(a'|s') mu(s',a')
        Then returns sum_s0 sum_a P(s0) P(a|s0) mu(s0,a).
         
        Assumes sup_{ (s,a) } ||phi(s,a)||_{\infty} <= 1.0
        '''
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
        
        # result = sum_{s} sum_{a} P(s)*P(a|s)*mu(s,a)
        for s in initial:
            pi = agent.actions(s)
            for a in pi:
                result += initial[s]*pi[a]*mu[ (s,a) ]
        return result
    
    def feature_expectations2(self, model, initial, agent):
        '''
        compute 
            mu = Phi*inv(I - gamma*P)
        where 
            P(i,j)     = P(s'=i|s=j,pi) = E_{a}[ P(s'=i|s=j,a) ] = sum_{a} P(a|s) P(s'=i|s=j,a)
            Phi(:,j)   = phi(s=j) = E_{a}[ phi(s=j,a) ] = sum_{a} P(a|s) phi(s,a)
            mu(:,j)    = mu(pi)(s=j) = E[sum_t gamma^t phi(s_t,a_t) | s_0=s]
        assumes agent chooses policy deterministically.
        '''
        # Index states
        S = {}
        for (i,s) in enumerate( model.S() ):
            S[s] = i
        
        #Initialize matrices
        ff = model.reward_function
        k = ff.dim
        n_S = len(S)
        Phi = np.zeros( (k,n_S) )
        P = np.zeros( (n_S, n_S) )
        
        # build Phi
        for s in S:
            Ta = agent.actions(s)
            for a in Ta:
                j = S[s]
                Phi[:,j] += ff.features(s,a)*Ta[a]
        
        # Build P
        for s in S:
            Ta = agent.actions(s)
            for a in Ta:
                Ts = model.T(s,a)
                for s_p in Ts:
                    i = S[s_p]
                    j = S[s]
                    P[i,j] += Ts[s_p]*Ta[a]
        # Calculate mu
        mu = np.dot( Phi, np.linalg.pinv(np.eye(n_S) - model.gamma*P) )
        
        # Calculate E_{s0}[ phi(s) ]
        result = np.zeros( k )
        for s in initial:
            j = S[s]
            result += initial[s]*mu[:,j]
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
        
        model: an MDP with a linear reward functions.  Parameters WILL be overwritten.
        initial: initial distribution over states
        samples: a list of sample trajectories [ (s_t,a_t) ] of the (supposedly) optimal
            policy.
        '''
        # Initial weight vector
        w_0 = model.feature_function.params
        
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
            
            print 'IRLApproxSolver Iteration #{},t = {:4.4f}'.format(i,t)
            
            # Compute optimal policy used R(s,a) = dot( feature_f(s,a), w^(i) )
            agent = self._solver.solve(model)
            
            # Compute feature expectations of pi^(i) = mu^(i)
            samples = self.generate_samples(model, agent, initial)
            mu = self.feature_expectations(model, samples)
            
        # Restore initial weight vector
        model.feature_function.params = w_0
        return (agent, w)
    
    def generate_samples(self, model, agent, initial):
        '''
        Generate self.n_samples different histories of length t_max by
        following agent.  Each history of the form,
            [ (s_0,a_0), (s_1,a_1), ...]
        '''
        # t_max such that gamma^t_max = 0.01
        t_max = math.ceil( math.log(0.01)/math.log(model.gamma) )
        result = []
        for i in range(self._n_samples):
            hist = []
            for (s,a,r,s_p) in mdp.simulation.simulate(model, agent, initial, t_max):
                hist.append( (s,a) )
            result.append(hist)
        return result
            
    def feature_expectations(self, model, samples):
        '''
        Compute empirical feature expectations
        E[sum_t gamma^t phi(s_t,a_t)] ~~ (1/m) sum_i sum_t gamma^t phi(s^i_t, a^i_t)
        '''
        ff = model.reward_function
        result = numpy.zeros( ff.dim )
        for sample in samples:
            for (t,sa) in enumerate(sample):
                s = sa[0]
                a = sa[1]
                result += (model.gamma**t)*ff.features(s,a)
        return (1.0/len(samples))*result
    
class ValueIteration(MDPSolver):
    '''
    Solve a model by repeated applications of the Bellman Operator
    '''
    def __init__(self, max_iter):
        self._max_iter = max_iter
        
    def solve(self, model):
        V = util.classes.NumMap()
        for i in range(self._max_iter):
            V = self.iter(model, V, 'max')
            print 'Value Iteration, Iter#{}'.format(i)
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

class QValueIteration(MDPSolver):
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
            Q = R + gamma*P*PI*Q
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
        
        # Solve Q = R + gamma*P*PI*Q
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
        
            
class PolicyIteration(MDPSolver):
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
            
            print 'PI Iteration #{}'.format(i)
            
            # Do a 1-step lookahead, choose best action
            policy = mdp.etc.policy_improvement(model, V_pi)
            new_agent = mdp.agent.MapAgent(policy)
            
            # Check if no improvement left
            n_different = len(mdp.etc.policy_difference(model.S(), agent, new_agent))
            if n_different == 0:
                print 'Policy Improvement renders no change'
                break
            
            # change agents
            agent = new_agent         
        return agent
        
class LSPI(MDPSolver):
    '''Least Squares Policy Iteration (Lagoudakis, Parr 2001)'''
    
    def __init__(self, n_iter, n_samples, feature_f=None):
        self._n_iter = n_iter
        self._n_samples = n_samples
        self._feature_f = feature_f
        
    def solve(self, model):
        '''
        Approximate Q^{\pi} = \Psi*w using weighted Least Squares
        based on samples
        '''
        if self._feature_f == None:
            feature_f = mdp.etc.StateActionFeatureFunction(model)
        else:
            feature_f = self._feature_f
        
        # initialize policy randomly
        agent = mdp.agent.RandomAgent(model.A())
        
        # Generate samples [ (s_0,a_0,r_0,s'_0), (s_1,a_1,r_1,s'_1),... ]
        initial = util.classes.NumMap()
        for s in model.S():
            initial[s] = 1
        initial = initial.normalize()
        samples = mdp.simulation.sample_model(model, self._n_samples, initial, agent)
        ## Does the a in (s,a,r,s') need to follow pi(s)?  
        ##     NO! lspi-short.pdf, 3/4 down page 4
        ## If deterministic policy used, what about missing (s,a) 
        ## for a != pi(s), equivalent to zero in stationary distribution 
        ## of (s,a)~MDP|pi?
        
        for i in range(self._n_iter):
            print 'LSPI Iteration #{}'.format(i)
            # evaluate policy approximately
#            w = self.lstdq_approx(samples, feature_f, model.gamma, agent)
            ## Is this correct?  implement exactly first
            w = self.lstdq_exact(samples, model, agent, feature_f)
            
            # Define an agent to use argmax over Q(s,a) to choose actions
            new_agent = mdp.etc.LinearQValueAgent(w,feature_f, model.A())
            
            # Check if no improvement left
            n_different = len(mdp.etc.policy_difference(model.S(), agent, new_agent))
            if n_different == 0:
                print 'Policy Improvement renders no change'
                break
            
            # change agents
            agent = new_agent   
             
        return mdp.etc.LinearQValueAgent(w,feature_f, model.A())
    
    def lstdq_approx(self, samples, feature_f, gamma, agent):
        '''
        Find weights to approximate value function of a given policy
            Q(s,a) ~~ w'*phi(s,a)
        
        '''
        k = feature_f.dim
        A = numpy.zeros( [k,k] )
        b = numpy.zeros( k )
        for (s,a,r,s_prime) in samples:
            phi = feature_f.features(s,a)
            phi_prime = feature_f.features(s_prime, agent.sample(s_prime))
            A = A + numpy.outer(phi,phi - gamma*phi_prime)
            b = b + phi*r
        w = numpy.dot( numpy.linalg.pinv(A), b )
        return w
    
    def lstdq_exact(self, samples, model, agent, feature_f):
        '''
        Return w such that
            Phi*w = R + gamma*P*Phi*w
        by solving
            w = inv(A)*b
        where
            A = Phi'*Delta*(Phi-gamma*P*Phi)
            b = Delta*Phi*R
            Phi[i,:] = Phi( (s,a)=i )
            P[i,j] = P( (s',a')=j | (s,a) = i )
            R[i] = R( (s,a)=i )
            Delta[i,i] = P( (s,a)=i ) in samples
        '''
        # Make index
        k = feature_f.dim
        SA = {}
        for (i,sa) in enumerate( itertools.product( model.S(), model.A() ) ):
            SA[sa] = i
        n_SA = len(SA)
            
        # initialize matrices
        Phi = np.zeros( (n_SA, k) )
        Delta = np.zeros( (n_SA, n_SA) )
        P = np.zeros( (n_SA, n_SA) )
        R = np.zeros( n_SA )
        
        # Feature Matrix
        for sa in SA:
            i = SA[sa]
            Phi[ i,: ] = feature_f.features(*sa)
        # Transition Matrix
        for sa in SA:
            Ts = model.T(*sa)
            for s_p in Ts:
                Ta = agent.actions(s_p)
                for a_p in Ta:
                    sa_p = (s_p,a_p)
                    j = SA[ sa_p ]
                    i = SA[ sa ]
                    P[i,j] = Ta[a_p]*Ts[s_p]
        # Weighting Matrix
        delta = util.classes.NumMap()
        for (s,a,r,s_p) in samples:
            delta[ (s,a) ] += 1
        delta = delta.normalize()
        for sa in SA:
            i = SA[sa]
            Delta[i,i] = delta[sa]
        # Reward Vector
        for sa in SA:
            i = SA[sa]
            R[i] = model.R(*sa)
        
        A = np.dot( np.dot( Phi.T, Delta), Phi-model.gamma*np.dot(Phi,P) )
        b = np.dot( np.dot( Phi.T, Delta), R )
        return np.dot( np.linalg.pinv(A), b)
    
    def generate_samples(self, model):
        '''
        Generate (s,a,r,s') pairs according to uniform distribution
        '''
        states = model.S()
        result = []
        for i in range(self._n_samples):
            s = random.sample(states,1)[0]
            a = random.sample(model.A(s),1)[0]
            r = model.R(s,a)
            s_p = util.functions.sample( model.T(s,a) )
            result.append( (s,a,r,s_p) )
        return result

class LSTD(MDPSolver):
    def __init__(self, n_iter=20, n_samples=1000, feature_f=None):
        self._n_iter = n_iter
        self._n_samples = n_samples
        self._feature_f = feature_f
    
    def solve(self, model):
        '''
        Use samples to solve estimate V = Psi*w using Least Squares
        '''
        if self._feature_f == None:
            feature_f = mdp.etc.StateFeatureFunction(model)
        else:
            feature_f = self._feature_f
        
        # initialize policy randomly
        agent = mdp.agent.RandomAgent(model.A())
        
        # Generate samples [ (s,a,r,s') ]
        initial = util.classes.NumMap()
        for s in model.S():
            initial[s] = 1
        initial = initial.normalize()
        samples = mdp.simulation.sample_model(model, self._n_samples, initial, agent)
        
        for i in range(self._n_iter):
            print 'LSTD Iteration #{}'.format(i)
            # Approximate Value of policy
            V_pi = self.lstd_exact(samples, model, feature_f, model.gamma, agent)
#            V_pi = self.lstd_approx(samples, model, feature_f, model.gamma, agent)
            
            # Improve on policy
            policy = mdp.etc.policy_improvement(model, V_pi)
            new_agent = mdp.agent.MapAgent(policy)
            
            # Check if nothing changes
            n_different = len(mdp.etc.policy_difference(model.S(), agent, new_agent))
            if n_different == 0:
                print 'Policy Improvement renders no change'
                break
            
            # change agent
            agent = new_agent
        return agent
    
    def lstd_approx(self, samples, model, feature_f, gamma, agent):
        '''
        Use least squares to estimate V^{\pi}(s) with phi(s)'*w.  Assumes
        agent is deterministic.
        
        Returns
            V^{\pi}(s)
        '''
        k = feature_f.dim
        A = np.zeros( (k,k) )
        b = np.zeros( k )
        for (s,a,r,s_p) in samples:
            phi = feature_f.features( s )
            phi_p = feature_f.features( s_p )
            A += np.outer(phi, phi - gamma*phi_p)
            b += r*phi
        w = np.dot( np.linalg.pinv(A), b)
        V = util.classes.NumMap()
        for s in model.S():
            V[s] = np.dot( feature_f.features(s), w)
        return V
    
    def lstd_exact(self, samples, model, feature_f, gamma, agent):
        '''
        Use least squares to estimate V^{\pi} ~~ Phi*w with
            w = inv(A)*b
        where
            A = Phi'*Delta*(Phi-gamma*P*Phi)
            b = Phi'*R
            Phi[i,:] = phi(s=i)
            Delta[i,i] = P(s=i) according to stationary distr of samples
            P[i,j] = P(s'=j|s=i) = sum_a P(a|s) P(s'|s,a)
            R = E[R(s)] = sum_a P(a|s) R(s,a)
        '''
        k = feature_f.dim
        
        # Make index
        S = {}
        for (i,s) in enumerate(model.S()):
            S[s] = i
        n_S = len(S)
        
        # Initialize Matrices
        Phi = np.zeros( (n_S,k) )
        Delta = np.zeros( (n_S,n_S) )
        P = np.zeros( (n_S, n_S) )
        R = np.zeros( n_S )
        
        # Phi
        for s in S:
            i = S[s]
            Phi[i,:] = feature_f.features(s)
        # P
        for s in S:
            Ta = agent.actions(s)
            for a in model.A(s):
                Ts = model.T(s,a)
                for s_p in Ts:
                    i = S[s]
                    j = S[s_p]
                    P[i,j] += Ta[a]*Ts[s_p]
        # R
        for s in S:
            Ta = agent.actions(s)
            for a in Ta:
                i = S[s]
                R[i] += model.R(s,a)*Ta[a]
        # Delta
        delta = util.classes.NumMap()
        for (s,a,r,s_p) in samples:
            delta[s] += 1
        delta = delta.normalize()
        for s in delta:
            i = S[s]
            Delta[i,i] = delta[s]
        
        # Actual computation
        A = np.dot( np.dot( Phi.T,Delta ), Phi - gamma*np.dot(P,Phi) )
        b = np.dot( Phi.T, np.dot(Delta, R) )
        w = np.dot( np.linalg.pinv(A),b )
        
        V = util.classes.NumMap()
        for s in model.S():
            V[s] = np.dot( feature_f.features(s), w)
        return V
        
            
            