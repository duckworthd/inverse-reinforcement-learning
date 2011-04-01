from model import *
from reward import *
from numpy import array
import numpy.random
import random
from mdp import agent, simulation
from util.NumMap import NumMap

if __name__ == '__main__':
    random.seed(0)
    numpy.random.seed(0)
    
    map_size = array( [5,5] )
    box_size = array( [2,2] )
    p_fail = 0.1
    initial = NumMap({GWState( (0,0) ):1.0})
    t_max = 10
    
    reward = GWBoxReward(box_size, map_size)
    reward_weights = numpy.random.rand( reward.dim() )
    reward.params = reward_weights
    
    model = GWModel(p_fail, map_size)
    model.reward_function = reward
    
    print reward
    print [str(state) for state in model.S()]
    print [str(action) for action in model.A()]
    
    agent = agent.HumanAgent(model)
    
    simulation.simulate(model, agent, initial, t_max)
    