from model import *
from reward import *
from numpy import array
import numpy.random
import random
from mdp import agent, simulation
from util.classes import NumMap
import mdp.solvers

if __name__ == '__main__':
    random.seed(0)
    numpy.random.seed(0)
    
    map_size = array( [5,5] )
    box_size = array( [2,2] )
    p_fail = 0.2
    initial = NumMap( {GWState( array( [0,0] ) ):1.0} )
    t_max = 20
    
    reward = GWBoxReward(box_size, map_size)
    reward_weights = numpy.random.rand( reward.dim )
    reward_weights[-5:] = 0.1*reward_weights[-5:]
    reward.params = reward_weights
    
    model = GWModel(p_fail, map_size)
    model.reward_function = reward
    model.gamma = 0.99
    
    print reward
    print [str(state) for state in model.S()]
    print [str(action) for action in model.A()]
    
#    agent = agent.HumanAgent(model)
#    agent = mdp.solvers.ValueIterator(100).solve(model)
    agent = mdp.solvers.LSPI(100,100).solve(model, initial)
    
    results = simulation.simulate(model, agent, initial, t_max)
    for (s,a,r) in results:
        print (str(s),str(a),r)
    