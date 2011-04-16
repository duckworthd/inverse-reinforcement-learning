'''
Created on Apr 16, 2011

@author: duckworthd
'''
import random
import numpy as np
import vacuumworld.model as vwmodel
import mdp.agent
import mdp.simulation
import mdp.solvers
import util.classes


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    
    ## Initialize constants
    map = np.array( [[1, 1, 1], 
                     [1, 0, 1]])
    p_fail = 0.2
    p_dust = 0.05
    start_state = vwmodel.VWState( np.array( [0,0] ), 
                                   np.array( [[1, 1, 0], 
                                              [0, 0, 1]] )
                                   )
    initial = util.classes.NumMap( {start_state:1.0} )
    t_max = 30
    
    ## Initialize model
    model = vwmodel.VWModel(map, p_fail, p_dust)
    model.gamma = 0.9
    
    ## Define player
#    policy = mdp.agent.HumanAgent(model)
#    policy = mdp.agent.RandomAgent(model.A())
    policy = mdp.solvers.ValueIterator(20).solve(model)
    
    ## Print
    print model.info()
    
    ## Simulate
    print 'Sample run:'
    for (s,a,r) in mdp.simulation.simulate(model, policy, initial, t_max):
        print '%s, %s, %f' % (s,a,r)
    