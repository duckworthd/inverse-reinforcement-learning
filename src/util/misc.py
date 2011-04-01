'''
Miscellaneous utility functions
'''
import random

def sample(distr):
    '''
    sample from a discrete probability distribution
    '''
    
    if len(distr) == 0:
        raise Exception("Can't sample empty distribution")
    v = random.random()
    total = 0
    for choice in distr.keys():
        total += distr[choice]
        if total >= v:
            return choice
    raise Exception("Sum of probability < 1.  Did you normalize?")