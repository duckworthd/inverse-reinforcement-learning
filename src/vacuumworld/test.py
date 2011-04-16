import unittest
import numpy as np
import vacuumworld.model as vwmodel
import vacuumworld.reward as vwreward
import util.functions

class VacuumWorldTest(unittest.TestCase):
    def setUp(self):
        self._map1 = np.array( [[1, 0, 1],
                                [1, 1, 1]] )
        self._map2 = np.array( [[1, 1, 1, 1, 1], 
                                [1, 0, 0, 1, 1],
                                [1, 0, 1, 1, 1],
                                [1, 1, 1, 0, 0]])
        self._act_fail = 0.2
        self._dust_prob = 0.05
        self._model1 = vwmodel.VWModel(self._map1, self._act_fail, self._dust_prob)
        self._model2 = vwmodel.VWModel(self._map2, self._act_fail, self._dust_prob)
    
    def test_state(self):
        s1 = vwmodel.VWState( np.array([3,2]), np.array([[1, 0, 1],
                                                         [0, 0, 0],
                                                         [1, 0, 1]]) )
        s2 = vwmodel.VWState( np.array([3,1]), np.array([[1, 0, 1],
                                                         [0, 0, 0],
                                                         [1, 0, 1]]) )
        s3 = vwmodel.VWState( np.array([3,2]), np.array([[1, 0, 1],
                                                         [0, 1, 0],
                                                         [1, 0, 1]]) )
        s4 = vwmodel.VWState( np.array([3,2]), np.array([[1, 0, 1, 0],
                                                         [0, 1, 0, 0],
                                                         [1, 0, 1, 0]]) )
        
        self.assertNotEqual(s1, s2)
        self.assertNotEqual(s2, s3)
        self.assertNotEqual(s1, s3)
        self.assertNotEqual(s1, s4)
        self.assertEqual(s1,s1)
    
    def test_move(self):
        s1 = vwmodel.VWState( np.array([3,2]), np.array([[1, 0, 1],
                                                         [0, 0, 0],
                                                         [1, 0, 1]]) )
        s2 = vwmodel.VWState( np.array([2,0]), np.array([[1, 0, 1],
                                                         [0, 0, 0],
                                                         [1, 0, 1]]) )
        s3 = vwmodel.VWState( np.array([7,5]), np.array([[1, 0, 1],
                                                         [0, 0, 0],
                                                         [1, 0, 1]]) )
        a1 = vwmodel.VWMoveAction( np.array( [-1,-2] ))
        a2 = vwmodel.VWMoveAction( np.array( [ 5, 5] ))
        
        self.assertEqual(s2, a1.apply(s1))
        self.assertEqual(s3, a2.apply(a1.apply(s1)))        
    
    def test_suck(self):
        s1 = vwmodel.VWState( np.array([2,2]), np.array([[1, 0, 1],
                                                         [0, 0, 0],
                                                         [1, 0, 1]]) )
        s2 = vwmodel.VWState( np.array([2,2]), np.array([[1, 0, 1],
                                                         [0, 0, 0],
                                                         [1, 0, 0]]) )
        a = vwmodel.VWSuckAction()
        self.assertEqual(s2, a.apply(s1))
    
    def test_S(self):
        s1 = self._map1.shape
        m1 = self._model1
        self.assertTrue(len(m1.S()) == 5*2**5)
        for loc in np.transpose( np.nonzero( np.ones( s1 ) )):
            for layout in util.functions.bitstrings( np.prod(s1) ):
                inds = np.transpose( np.nonzero( np.ones( s1 ) ))
                s = vwmodel.VWState( loc, util.functions.sparse_matrix(s1, inds, layout))
                if m1.is_legal(s):
                    self.assertTrue(s in m1.S())
    
    def test_is_legal(self):
        s1 = vwmodel.VWState( np.array([0,0]), np.array([[1, 0, 1],
                                                         [0, 0, 0]]) )
        s2 = vwmodel.VWState( np.array([1,1]), np.array([[1, 1, 1],
                                                         [0, 0, 0]]) )
        s3 = vwmodel.VWState( np.array([0,0]), np.array([[1, 1, 1],
                                                         [0, 0, 0]]) )
        s4 = vwmodel.VWState(np.array([-1,0]), np.array([[1, 0, 1],
                                                         [1, 0, 0]]) )
        s5 = vwmodel.VWState( np.array([0,1]), np.array([[1, 0, 1],
                                                         [0, 0, 1]]) )
        m = self._model1
        self.assertTrue(m.is_legal(s1))
        self.assertFalse(m.is_legal(s2))    # invalid dust location
        self.assertFalse(m.is_legal(s3))    # invalid dust location 2
        self.assertFalse(m.is_legal(s4))    # invalid robot location
        self.assertFalse(m.is_legal(s5))    # invalid robot location 2
    
    def test_T(self):
        # Test a move action.  Probabilities calculated by hand.
        q = self._dust_prob
        p = self._act_fail 
        s = vwmodel.VWState( np.array([1,1]), np.array([[1, 0, 0],
                                                        [0, 1, 0]]) )
        a = vwmodel.VWMoveAction( np.array([0,1]) )
        T = self._model1.T(s,a)
        act_prob = {(1,2):(1-p), (1,1):2*p/3, (1,0):p/3 }   # probability of robot's location
        for loc in [ (1,2), (1,1), (1,0) ]:
            for layout in util.functions.bitstrings(3):
                inds = [(1,0), (1,2), (0,2)]
                dust = np.array( s.dust )   # copy current state's dust
                dust = util.functions.sparse_matrix( (2,3), inds, layout, dust )    # update dust
                n_dust = np.sum(dust)       # numbero of dust on the map
                s_p = vwmodel.VWState(np.array(loc), dust)
                dust_prob = ( q**(n_dust-2) )*( (1-q)**(5-n_dust) ) # probability of dust layout
                self.assertTrue(np.abs( T[s_p] - act_prob[loc]*dust_prob) < 1e-10)
                
        # Test a suck Action
        a = vwmodel.VWSuckAction()
        T = self._model1.T(s,a)
        for layout in util.functions.bitstrings(3):
            inds = [(1,0), (1,2), (0,2)]
            dust = np.array( s.dust )
            dust = util.functions.sparse_matrix( (2,3), inds, layout, dust )
            n_dust = np.sum(dust)
            dust[1,1] = 0
            s_p = vwmodel.VWState(s.robot, dust)
            dust_prob = ( q**(n_dust-2) )*( (1-q)**(5-n_dust) )
            self.assertTrue(np.abs( T[s_p] - dust_prob) < 1e-10)
            
        for s in self._model1.S():
            for a in self._model1.A():
                T = self._model1.T(s,a)
                self.assertTrue( abs( sum(T.values())-1.0 ) < 10e-10 )
            
    def test_reward(self):
        reward = vwreward.VWReward()
        dust = np.array([[1, 0],
                         [1, 1]])
        s = vwmodel.VWState( np.array( [1,1] ), dust)
        a = vwmodel.VWSuckAction()
        
        self.assertEqual(reward.reward(s, a), -3)
        
        a = vwmodel.VWMoveAction( np.array( [-1,0] ) )
        
        self.assertEqual(reward.reward(s,a), -4)
        
    
if __name__ == '__main__':
    unittest.main()