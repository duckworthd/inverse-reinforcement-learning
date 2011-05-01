'''
Created on Apr 16, 2011

@author: duckworthd
'''
import random
import numpy as np
import vacuumworld.model as vwmodel
import vacuumworld.reward as vwreward
import vacuumworld.etc as vwetc
import mdp.agent
import mdp.simulation
import mdp.solvers
import util.classes

def vacuum_main():
    random.seed(0)
    np.random.seed(0)
    
    ## Initialize constants
#    map = np.array( [[1, 1, 1, 1], 
#                     [1, 0, 1, 1],
#                     [1, 1, 0, 1]])
    map = np.array( [[1, 1, 1], 
                     [1, 0, 1]])
    p_fail = 0.2
    p_dust = 0.05
#    start_state = vwmodel.VWState( np.array( [0,0] ), 
#                                   np.array( [[1, 1, 0, 0], 
#                                              [0, 0, 1, 0],
#                                              [0, 0, 0, 0]] )
#                                   )
    start_state = vwmodel.VWState( np.array( [0,0] ), 
                                   np.array( [[1, 1, 0], 
                                              [0, 0, 1]] )
                                   )
    initial = util.classes.NumMap( {start_state:1.0} )
    t_max = 500
    
    ## Initialize model
    model = vwmodel.VWModel(map, p_fail, p_dust)
    model.gamma = 0.99
    model.reward_function = vwreward.VWLinearReward(map)
    
    ## Define player
#    policy = mdp.agent.HumanAgent(model)
#    policy = mdp.agent.RandomAgent(model.A())
    opt_policy = mdp.solvers.ValueIteration(100).solve(model)
#    policy = mdp.solvers.PolicyIterator(20, mdp.solvers.ExactPolicyEvaluator()).solve(model)
#    policy = mdp.solvers.PolicyIterator(20, mdp.solvers.IteratingPolicyEvaluator(100)).solve(model)
#    policy = mdp.solvers.PolicyIterator(20, mdp.solvers.SamplingPolicyEvaluator(100, 50)).solve(model)
    policy = mdp.solvers.LSPI(50, 5000, vwetc.VWFeatureFunction()).solve(model)
#    policy = mdp.solvers.LSPI(50, 5000).solve(model)
    
    ## Print
    print model.info()
    n_different = 0
    for s in model.S():
        if opt_policy.actions(s) != policy.actions(s):
            n_different += 1
    print 'Optimal Policy and Approx Policy differ on {} states of {}'.format(n_different, len(model.S()))
    
    ## Simulate
    print 'Sample run:'
    for (s,a,r) in mdp.simulation.simulate(model, policy, initial, t_max):
        print '%s, %s, %f' % (s,a,r)
    

if __name__ == '__main__':
    vacuum_main()