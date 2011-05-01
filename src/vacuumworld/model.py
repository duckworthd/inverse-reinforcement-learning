import mdp.model
import vacuumworld.reward
import numpy as np
import util.functions
import util.classes

class VWState(mdp.model.State):
    '''
    Robot Location + Dust Locations
    '''
    def __init__(self, robot, dust):
        '''
        robot = robot location as an ndarray
        dust = matrix of same size as map where
                dust[i,j] = 1 if (i,j) is dirty, 0 else
        '''
        self.robot = robot
        self.dust = dust
        
#    def __str__(self):
#        return 'VWState [robot={}, dust={}'.format(self.robot, self.dust)
    
    def __eq__(self, other):
        try:
            return np.all( self.robot == other.robot ) and \
                np.all( self.dust == other.dust )
        except Exception:
            return False
    
    def __hash__(self):
        result = hash( tuple( self.robot ) )
        for row in np.nonzero(self.dust):
            result += hash( tuple(row) )
        return result
    
    def __str__(self):
        result = ['VWState:\n']
        (m,n) = self.dust.shape
        for i in reversed(range(m)):
            for j in range(n):
                if self.dust[i,j] == 1:
                    result.append( '[O]')
                else:
                    result.append( '[-]')
                if np.all( self.robot == np.array( (i,j) ) ):
                    result[-1] = '[X]'
            result.append( '\n' )
        return ''.join(result)

class VWMoveAction(mdp.model.Action):
    '''
    Action that moves the robot
    '''
    def __init__(self, direction):
        self.direction = direction
        
    def apply(self, s):
        new_loc = s.robot + self.direction
        return VWState( new_loc, np.array(s.dust) )
    
    def __str__(self):
        return 'VWMoveAction: [direction={}'.format(self.direction)
    
    def __eq__(self, other):
        try:
            return np.all(self.direction == other.direction)
        except Exception:
            return False
    
    def __hash__(self):
        return hash( tuple( self.direction ) )
    
class VWSuckAction(mdp.model.Action):
    '''
    Action that sucks dirt out from under the robot
    '''
    def apply(self, s):
        new_dust = np.array(s.dust)
        new_dust[ tuple(s.robot) ] = 0
        return VWState( np.array(s.robot), new_dust )
    
    def __str__(self):
        return 'VWSuckAction'
    
    def __eq__(self, other):
        return isinstance(other, VWSuckAction)
    
    def __hash__(self):
        return 255

class VWModel(mdp.model.Model):
    """
    A Vacuum world where a single robot wanders around picking up dust
    that appears randomly throughout time. 
    """
    def __init__(self, map=np.ones([4,3]), p_fail=0.2, p_dust=0.01):
        '''
        map: a boolean 2D ndarray with 1's for locations the robot can be at
            and 0s otherwise
        p_fail: the probability that a move action will fail.  Probability mass of
            p_fail will be split equally between all move actions other than the one
            intended
        p_dust: probability that each tile will become dusty in a time step.  Each
            tile treated independently.
        '''
        self._map = map
        self._p_fail = p_fail
        self._p_dust = p_dust
        
        up      = np.array( [0,1] )
        left    = np.array( [-1,0])
        right   = np.array( [1,0] )
        down    = np.array( [0,-1])
        
        self._move_actions = [VWMoveAction(up), VWMoveAction(left),
                         VWMoveAction(right), VWMoveAction(down)]
        self.reward_function = vacuumworld.reward.VWReward()
    
    def T(self,state,action):
        """Returns a function state -> [0,1] for probability of next state
        given currently in state performing action"""
        valid_locs = [ tuple(i) for i in np.transpose( np.nonzero(self._map)) ]
        
        # Robot movement transitions
        robot_locs = util.classes.NumMap()  #Uses states, but only robot location relevant
        if action in self._move_actions:
            # Move Action.  Dust layout at s' independent of action.
            for a in self._move_actions:
                p = 0
                if a == action:
                    p = 1 - self._p_fail
                else:
                    p = self._p_fail / ( len(self._move_actions)-1 )
                s_p = a.apply(state)
                if not self.is_legal(s_p):
                    robot_locs[state] += p
                else:
                    robot_locs[s_p] += p
        else:
            # Suck Action.  Robot doesn't move, dust layout NOT independent
            # of action.
            robot_locs[state] = 1.0
        
        # all possible dust assignment transitions
        no_dust = []    # all positions that COULD become dust
        for loc in valid_locs:
            if state.dust[loc] == 0:
                no_dust.append(loc)
        dust_locs = util.classes.NumMap()   #Uses states, but only dust locs relevant
        if len(no_dust) > 0:
            for dust_layout in util.functions.bitstrings( len(no_dust) ):
                new_dust = util.functions.sparse_matrix(self._map.shape, 
                                                        no_dust, dust_layout, 
                                                        np.array( state.dust ))
                # p(layout) = p_dust^{# of new dust introduced) * 
                #            (1-p_dust)^{# of places dust didn't appear}
                dust_locs[VWState(state.robot, new_dust)] = ( self._p_dust**np.sum(dust_layout) ) *\
                    ( (1-self._p_dust)**(len(dust_layout)-np.sum(dust_layout)) )
        else:
            dust_locs[VWState( state.robot, np.array(state.dust) )] = 1.0
        
        # combine
        result = util.classes.NumMap()
        for (robot_loc, r_p) in robot_locs.items():
            for (dust_loc, d_p) in dust_locs.items():
                if action == VWSuckAction():
                    # Remove dust under agent if action was Suck
                    dust_loc = VWSuckAction().apply(dust_loc)
                result[VWState(robot_loc.robot, dust_loc.dust)] += r_p*d_p
        return result
        
    def S(self):
        """All states in the MDP"""
        result = []
        size = self._map.shape
        inds = np.transpose( np.nonzero( self._map ) )  # all legal positions in self._map
        for dust_layout in util.functions.bitstrings( len(inds) ):
            for ind in inds:
                result.append( VWState(ind, util.functions.sparse_matrix(size, inds, dust_layout)))
        return result
        
    def A(self,state=None):
        """All actions in the MDP is state=None, otherwise actions available
        from state"""
        return self._move_actions + [VWSuckAction()]
    
    def is_terminal(self, state):
        '''returns whether or not a state is terminal'''
        return False
    
    def is_legal(self, state):
        # Check robot location
        robot = state.robot
        if not ( np.all(robot >= 0) and np.all( robot < self._map.shape ) and 
                 self._map[ tuple(robot) ] == 1 ):
            return False
        
        # Check dust locations
        dust = state.dust
        if not np.all(dust.shape == self._map.shape):
            return False
        for pos in np.transpose( np.nonzero( dust )):
            if self._map[ tuple(pos) ] == 0:
                return False
        return True
    
    def info(self):
        result = ['VWModel:\n']
        (m,n) = self._map.shape
        for i in reversed(range(m)):
            for j in range(n):
                if self._map[i,j] == 1:
                    result.append( '[O]')
                else:
                    result.append( '[1]')
            result.append( '\n' )
        return ''.join(result)
