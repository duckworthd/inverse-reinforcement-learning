'''
Created on Apr 25, 2011

@author: duckworthd
'''
import util.classes
import numpy as np
import vacuumworld.model as vwmodel

class VWFeatureFunction(util.classes.FeatureFunction):
    '''
    A feature function for vacuum world.  Includes
    -# of dirty tiles
    -L1 distance to closest dirty tile
    -move-action indicator
    -indicator if robot is over dust and attempting suck action selected
    '''
    @property
    def dim(self):
        '''dimension of all output from self.features()'''
        return 4
    
    def features(self, state, action):
        '''Calculate features for arguments.  Returns
        numpy.ndarray of length self.dim'''
        result = np.zeros( self.dim )
        
        # number of dirty tiles
        result[0] = np.sum(state.dust)
        
        # closest dirty tile distance in L1 norm
        robot = state.robot
        if result[0] == 0:
            result[1] = 0
        else:
            result[1] = np.sum( state.dust.shape )
            for dirty_tile in np.transpose( np.nonzero( state.dust ) ):
                result[1] = min(result[1], np.sum( np.abs(robot - dirty_tile )))
        
        # action type
        if isinstance(action, vwmodel.VWMoveAction):
            result[2] = 1
        elif isinstance(action, vwmodel.VWSuckAction) and state.dust[tuple(robot)] != 0:
            result[3] = 1
        
        return result