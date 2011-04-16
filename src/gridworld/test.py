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
            
        for s in self._model.S():
            for a in self._model.A():
                self.assertTrue( abs( sum(self._model.T(s,a).values())-1.0 ) < 1e-10 )
        
    
    def test_action(self):
        a = GWAction( np.array( [-1, 5] ) )
        s = GWState( np.array([5,5]) )
        s_p = a.apply(s)
        s_p_true = GWState( np.array([4,10]) )
        self.assertTrue(s_p == s_p_true, 'Action result incorrect')
        self.assertTrue( np.all(a.direction == np.array([-1,5])) )

if __name__ == '__main__':
    unittest.main()
