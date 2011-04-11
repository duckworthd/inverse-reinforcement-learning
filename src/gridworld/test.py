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

def evaluate_policy(model, initial, agent, t_max=100):
    scores = []
    for i in range(100):
        results = simulation.simulate(model, agent, initial, t_max)
        score = 0
        for (i, (s,a,r)) in enumerate(results):
            score += r*model.gamma**i
        scores.append(score)
    return sum(scores)/len(scores)

if __name__ == '__main__':
    random.seed(0)
    numpy.random.seed(0)
    
    ## Initialize constants
    map_size = array( [5,5] )
    box_size = array( [2,2] )
    p_fail = 0.2
    initial = NumMap( {GWState( array( [0,0] ) ):1.0} )
    t_max = 20
    
    ## Create reward function
    reward = GWBoxReward(box_size, map_size)
    reward_weights = numpy.random.rand( reward.dim )
    reward_weights[-5:] = 0.1*reward_weights[-5:]
    reward.params = reward_weights
    
    ## Create Model
    model = GWModel(p_fail, map_size)
    model.reward_function = reward
    model.gamma = 0.9
    
    ## Define feature function (approximate methods only)
    feature_function = mdp.etc.CompleteFeatureFunction(model)
#    feature_function = GWLocationFF(model)
    
    ## Define player
#    agent = agent.HumanAgent(model)
#    agent = mdp.solvers.ValueIterator(100).solve(model)
#    agent = mdp.solvers.PolicyIterator(20, mdp.solvers.ExactPolicyEvaluator(100)).solve(model)
    agent = mdp.solvers.PolicyIterator(20, mdp.solvers.ApproximatePolicyEvaluator(100,50)).solve(model)
#    agent = mdp.solvers.QValueIterator(100).solve(model)
#    agent = mdp.solvers.LSPI(20,1000).solve(model)
    
    ## Print out world information
    print reward
    print [str(state) for state in model.S()]
    print [str(action) for action in model.A()]
    print '\n'
    
    ## Estimate policy quality
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