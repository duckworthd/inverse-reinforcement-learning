'''
Created on Apr 9, 2011

@author: duckworthd
'''
import numpy as np
import util.classes
import mdp.agent

class StateActionFeatureFunction(util.classes.FeatureFunction):
    '''A feature function with indicators for every (s,a) pair'''
    def __init__(self,mdp):
        self._ind = util.classes.NumMap()
        for (i,s) in enumerate( mdp.S() ):
            for (j,a) in enumerate( mdp.A() ):
                self._ind[ (s,a) ] = i*len( mdp.A() ) + j
    @property
    def dim(self):
        '''dimension of all output from self.features()'''
        return len(self._ind)
    
    def features(self, s, a):
        '''Calculate features for arguments.  Returns
        numpy.ndarray of length self.dim'''
        result = np.zeros( self.dim )
        result[ self._ind[ (s,a) ] ] = 1.0
        return result
    
class StateFeatureFunction(util.classes.FeatureFunction):
    '''A feature function with indicators for every state'''
    def __init__(self,model):
        self._ind = util.classes.NumMap()
        for (i,s) in enumerate( model.S() ):
            self._ind[s] = i
            
    @property
    def dim(self):
        '''dimension of all output from self.features()'''
        return len(self._ind)
    
    def features(self, s):
        '''Calculate features for arguments.  Returns
        numpy.ndarray of length self.dim'''
        result = np.zeros( self.dim )
        result[ self._ind[ s ] ] = 1.0
        return result
    
class LinearQValueAgent(mdp.agent.Agent):
    '''Implicitly encodes a policy using an MDP's feature function
    and available actions for a given state along with its own weights'''
    def __init__(self, weights, feature_f, actions):
        self._w = weights
        self._ff = feature_f
        self._A = actions
    
    def actions(self,state):
        actions = util.classes.NumMap()
        for a in self._A:
            phi = self._ff.features(state, a)
            actions[a] = np.dot( phi, self._w )
        result = util.classes.NumMap()
        result[actions.argmax()] = 1.0
        return result
    
    
def policy_improvement(model, V):
    '''
    Do a 1-step lookahead and choose the best action for each state
    '''
    policy = {}
    for s in model.S():
        actions = util.classes.NumMap()
        for a in model.A(s):
            v = model.R(s,a)
            T = model.T(s,a)
            v += model.gamma*sum( [trans_prob*V[s_prime] for (s_prime,trans_prob) in T.items()] )
            actions[a] = v
        policy[s] = actions.argmax()
    return policy

def policy_difference(states, agent1, agent2):
    '''
    Return
        [s_0, s_1, s_2, ...]
    
    where for each state agent1.actions(s) != agent2.actions(s)
    '''
    return [s for s in states if agent1.actions(s) != agent2.actions(s)]

def eval_policy(optimal_policy, other_policy, evaluator, model):
    '''
    Evaluate a policy, ||V* - V^{\pi}||_{\infty}.
    Requires full state enumeration.
    '''
    V_opt = evaluator.evaluate_policy(model, optimal_policy)
    V_other = evaluator.evaluate_policy(model, other_policy)
    result = 0
    for s in model.S():
        result = max(result, abs(V_opt[s] - V_other[s]))
    return result

def eval_policy_loss_exact(optimal_policy, other_policy, evaluator, model, initial):
    '''
    Calcluate E[V*(s_{0}) - V_{est}(s_{0})] where 
        V*(s)       = E[\sum_{t} \gamma^{t} R(s_{t}, a_{t}) | pi*]
        V_{est}(s)  = E[\sum_{t} \gamma^{t} R(s_{t}, a_{t}) | pi_{est}]
        s_{0} sampled from initial
    '''
    V_opt = evaluator.evaluate_policy(model, optimal_policy)
    V_other = evaluator.evaluate_policy(model, other_policy)
    result = 0
    for s in initial:
        result += initial[s]*(V_opt[s] - V_other[s])
    return result

def eval_policy_loss_approx(model, initial, policy, t_max=100, n_samples=100):
    '''
    Calcluate E[V*(s_{0}-V_{est}(s_{0}] where 
        V*(s)       = E[\sum_{t} \gamma^{t} R(s_{t}, a_{t}) | pi*]
        V_{est}(s)  = E[\sum_{t} \gamma^{t} R(s_{t}, a_{t}) | pi_{est}]
        s_{0} sampled from initial
    using n_samples from initial and running for t_max steps
    '''
    scores = []
    for i in range(n_samples):
        results = mdp.simulation.simulate(model, policy, initial, t_max)
        score = 0
        for (i, (s,a,r)) in enumerate(results):
            score += r*model.gamma**i
        scores.append(score)
    return sum(scores)/len(scores)

def R_max(model):
    '''
    Calculate bound for reward.  |R(s,a)| <= R_max for all (s,a)
    '''
    result = 0
    for s in model.S():
        for a in model.A():
            result = max(result, abs(model.R(s,a)))
    return result

def policy_report(opt_policy, policy, evaluator, model, initial):
    '''
    Print out an evaluation of a policy compared to an optimal policy
    '''
    print '{:>30s}: {:4.4f}'.format('||V* - V_est||_{inf}', 
                                  eval_policy(opt_policy, policy, evaluator, model))
    print '{:>30s}: {:4.4f}'.format('E[V*(s_0)-V_est(s_0)]', 
                                  eval_policy_loss_exact(opt_policy, policy, evaluator, model, initial))
    print '{:>30s} <= {:4.4f}'.format('|V(s)|', R_max(model)/(1-model.gamma))