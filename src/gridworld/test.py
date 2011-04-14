from model import *
from reward import *
from etc import *
from numpy import array
import numpy.random
import random
from mdp import agent, simulation
from util.classes import NumMap
import mdp.solvers
import mdp.etc
import unittest

class TestGridworld(unittest.TestCase):
    def setUp(self):
        self._map = array( [[1, 1, 1, 1, 1], 
                            [1, 0, 0, 1, 1],
                            [1, 0, 1, 1, 1],
                            [1, 1, 1, 0, 0]])
        self._p_fail = 0.2
        self._terminal = GWState( np.array( [0,4] ) )
        self._model = GWModel(self._p_fail, self._map, self._terminal)
        
    def test_is_legal(self):
        model = self._model
        (r,c) = self._map.shape
        for i in range(r):
            for j in range(c):
                state = GWState( np.array([i,j]) )
                if self._map[i,j] == 1:
                    self.assertTrue(model.is_legal(state), 'Legal state believed illegal')
                else:
                    self.assertFalse(model.is_legal(state), 'Illegal state believed legal')
    
    def test_is_terminal(self):
        model = self._model
        (r,c) = self._map.shape
        for i in range(r):
            for j in range(c):
                state = GWState( np.array([i,j]) )
                if np.all( state.location == self._terminal.location ):
                    self.assertTrue(model.is_terminal(state), 'Terminal state believed nonterminal')
                else:
                    self.assertFalse(model.is_terminal(state), 'Nonterminal state believed terminal')
    
    def test_S(self):  
        S = self._model.S()
        for s in S:
            (i,j) = tuple(s.location)
            self.assertTrue(self._map[i,j] == 1, 'Illegal state appearing amongst legal states')
            
    def test_T(self):
        a = GWAction( np.array( [1,0] ) )
        s = GWState( np.array( [2,2] ) )
        T = self._model.T(s,a)
        pairs = [(GWState( np.array([3,2]) ), 0.8),
                 (GWState( np.array([2,3]) ), 0.2/3),
                 (GWState( np.array([2,2]) ), 2*(0.2/3))]
        for (s_p, t) in pairs:
            self.assertTrue( abs(T[s_p]-t) < 1e-10 ) 
        
    
    def test_action(self):
        a = GWAction( np.array( [-1, 5] ) )
        s = GWState( np.array([5,5]) )
        s_p = a.apply(s)
        s_p_true = GWState( np.array([4,10]) )
        self.assertTrue(s_p == s_p_true, 'Action result incorrect')
        self.assertTrue( np.all(a.direction == np.array([-1,5])) )
            
def evaluate_policy(model, initial, agent, t_max=100):
    '''Sample t_max runs of agent in model starting from initial'''
    scores = []
    for i in range(100):
        results = simulation.simulate(model, agent, initial, t_max)
        score = 0
        for (i, (s,a,r)) in enumerate(results):
            score += r*model.gamma**i
        scores.append(score)
    return sum(scores)/len(scores)

if __name__ == '__main__':
#    unittest.main()
    
    random.seed(0)
    numpy.random.seed(0)
    
    ## Initialize constants
    map = array( [[1, 1, 1, 1, 1], 
                  [1, 0, 0, 1, 1],
                  [1, 0, 1, 1, 1],
                  [1, 1, 1, 0, 0]])
    box_size = array( [2,2] )
    p_fail = 0.2
    initial = NumMap( {GWState( array( [0,0] ) ):1.0} )
    t_max = 20
    
    ## Create reward function
    reward = GWBoxReward(box_size, map)
    reward_weights = numpy.random.rand( reward.dim )
    reward_weights[-5:] = 0.1*reward_weights[-5:]
    reward.params = reward_weights
    
    ## Create Model
    model = GWModel(p_fail, map)
    model.reward_function = reward
    model.gamma = 0.9
    
    ## Define feature function (approximate methods only)
    feature_function = mdp.etc.CompleteFeatureFunction(model)
#    feature_function = GWLocationFF(model)
    
    ## Define player
#    agent = agent.HumanAgent(model)
    agent = mdp.solvers.ValueIterator(100).solve(model)
#    agent = mdp.solvers.QValueIterator(100).solve(model)
#    agent = mdp.solvers.LSPI(20,1000).solve(model)
#    agent = mdp.solvers.PolicyIterator(20, mdp.solvers.ExactPolicyEvaluator(100)).solve(model)
#    agent = mdp.solvers.PolicyIterator(20, mdp.solvers.ApproximatePolicyEvaluator(100,50)).solve(model)
    
    ## Print out world information
    print model.info()
    print reward.info()
    print 'States: ' + str( [str(state) for state in model.S()] )
    print 'Action: ' + str( [str(action) for action in model.A()] )
    print 'Policy: '
    for s in model.S():
        print '\tpi({}) = {}'.format(s, agent.actions(s))
    
    ## Estimate policy quality
    print 'Sample run:'
    for (s,a,r) in simulation.simulate(model, agent, initial, t_max):
        print '%s, %s, %f' % (s,a,r)
    print 'Average Score: %f' % (evaluate_policy(model, initial, agent, t_max),)
    
#    ## Do IRL
##    irl = mdp.solvers.IRLExactSolver(20, mdp.solvers.ValueIterator(100))
##    (estimated_agent, estimated_weights) = irl.solve(model, initial, agent) 
#    irl = mdp.solvers.IRLApprximateSolver(20, mdp.solvers.ValueIterator(100), 100)
#    samples = irl.generate_samples(model, agent, initial)
#    (estimated_agent, estimated_weights) = irl.solve(model, initial, samples) 
#    
#    ## Estimate estimated policy quality
#    print 'Average Score: %f' % (evaluate_policy(model, initial, estimated_agent, t_max),)
#    
#    for s in model.S():
#        print 's = %s, pi*(s) = %s, pi_E(s) = %s' % ( s, agent.sample(s), estimated_agent.sample(s) )