import gridworld.model
import gridworld.reward
import numpy as np
import random
import mdp.simulation
import mdp.solvers
#import mdp.agent
import util.classes
  
def evaluate_policy(model, initial, policy, t_max=100):
    '''Sample t_max runs of mdp.agent in model starting from initial'''
    scores = []
    for i in range(100):
        results = mdp.simulation.simulate(model, policy, initial, t_max)
        score = 0
        for (i, (s,a,r)) in enumerate(results):
            score += r*model.gamma**i
        scores.append(score)
    return sum(scores)/len(scores)

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    
    ## Initialize constants
    map = np.array( [[1, 1, 1, 1, 1], 
                  [1, 0, 0, 1, 1],
                  [1, 0, 1, 1, 1],
                  [1, 1, 1, 0, 0]])
    box_size = np.array( [2,2] )
    p_fail = 0.2
    initial = util.classes.NumMap( {gridworld.model.GWState( np.array( [0,0] ) ):1.0} )
    t_max = 50
    
    ## Create reward function
    reward = gridworld.reward.GWBoxReward(box_size, map)
    reward_weights = np.random.rand( reward.dim )
    reward_weights[-4:] = 0.1*reward_weights[-4:]
#    reward_weights[-4:] = np.zeros( 4 )
    reward.params = reward_weights
    
    ## Create Model
    model = gridworld.model.GWModel(p_fail, map)
    model.reward_function = reward
    model.gamma = 0.9
    
    ## Define feature function (approximate methods only)
    feature_function = mdp.etc.CompleteFeatureFunction(model)
#    feature_function = GWLocationFF(model)
    
    ## Define player
#    policy = mdp.agent.HumanAgent(model)
    policy = mdp.solvers.ValueIterator(100).solve(model)
#    policy = mdp.solvers.QValueIterator(100).solve(model)
#    policy = mdp.solvers.LSPI(40,1000).solve(model)
#    policy = mdp.solvers.PolicyIterator(20, mdp.solvers.ExactPolicyEvaluator(100)).solve(model)
#    policy = mdp.solvers.PolicyIterator(20, mdp.solvers.ApproximatePolicyEvaluator(100,50)).solve(model)
    
    ## Print out world information
    print model.info()
    print reward.info()
    print 'States: ' + str( [str(state) for state in model.S()] )
    print 'Action: ' + str( [str(action) for action in model.A()] )
    print 'Policy: '
    for s in model.S():
        print '\tpi({}) = {}'.format(s, policy.actions(s))
    
    ## Estimate policy quality
    print 'Sample run:'
    for (s,a,r) in mdp.simulation.simulate(model, policy, initial, t_max):
        print '%s, %s, %f' % (s,a,r)
    print 'Average Score: %f' % (evaluate_policy(model, initial, policy, t_max),)
    
    ## Do IRL
    irl = mdp.solvers.IRLExactSolver(20, mdp.solvers.ValueIterator(100))
    (estimated_policy, estimated_weights) = irl.solve(model, initial, policy) 
#    irl = mdp.solvers.IRLApprximateSolver(20, mdp.solvers.ValueIterator(100), 100)
#    samples = irl.generate_samples(model, policy, initial)
#    (estimated_policy, estimated_weights) = irl.solve(model, initial, samples) 
    
    ## Estimate estimated policy quality
    model.reward_function.params = reward_weights
    print 'Average Score: %f' % (evaluate_policy(model, initial, estimated_policy, t_max),)
    
    for s in model.S():
        print 's = %s, pi*(s) = %s, pi_E(s) = %s' % ( s, policy.actions(s), estimated_policy.actions(s) )
    print 'pi* and pi_E disagree on {} of {} states'.format( len([ s for s in model.S() if 
                                                            policy.actions(s) != estimated_policy.actions(s) ]),
                                                            len(model.S()) )