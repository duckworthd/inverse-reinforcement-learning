'''
Miscellaneous utility functions
'''
import random
import numpy as np

def sample(distr, n_samples=1):
    '''
    generate n_samples from a discrete probability distribution
    '''
    if len(distr) == 0:
        raise Exception("Can't sample empty distribution")
    v = random.random()
    total = 0
    for choice in distr.keys():
        total += distr[choice]
        if total >= v:
            return choice
    raise Exception( "Sum of probability = {} < 1.  Did you normalize? {}".format(sum(distr.values()),
                                                                                  distr) )
def samples(distr, n_samples):
    '''
    Generate n_sampels from distr were distr[event] = Pr(event)
    '''
    # Generate index
    index = []
    total = 0
    for (key, prob) in distr.items():
        total += prob
        index.append( (total, key) )
    if abs(1-index[-1][0]) > 1e-8:
        raise Exception( "Sum of probability = {} < 1.  Did you normalize? {}".format(index[-1][0],
                                                                                      distr) )
    # sample from index using TODO binary search
    result = []
    for i in range(n_samples):
        p = random.random()
        for (p_total, key) in index:
            if p_total >= p:
                result.append(key)
                break
    return result

def bitstrings(n):
    '''
    Returns all bit strings of length n as a list of ndarray objects.  
    If n=0, returns ['0']
    '''
    if n <= 0:
        return np.array([0],dtype='bool_')
    elif n == 1:
        return [np.array( [0],dtype='bool_' ), 
                np.array( [1],dtype='bool_' )]
    else:
        shorter = bitstrings(n-1)
        result = []
        for i in [np.array( [0], dtype='bool_'), np.array( [1], dtype='bool_')]:
            for j in shorter:
                result.append( np.hstack( (i,j) ) )
        return result
    
def sparse_matrix(size, indices, values, M=None):
    '''
    size = (d1,d2,d3,...), d1 = length of dimension 1, ...
    indices = [(a1,a2,a3,...), (b1,b2,b3,...)] indices of each nonzero entry
    values = values corresponding to indices
    M = matrix to fill with values. a new, all-zero matrix is filled if None.
        M.shape = size required.
    '''
    if M == None:
        M = np.zeros(size)
    for (i,ind) in enumerate(indices):
        M[ tuple(ind) ] = values[i]
    return M