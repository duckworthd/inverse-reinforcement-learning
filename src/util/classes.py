
class NumMap(dict):
    """A dict that explicitly maps to numbers"""
    
    def max(self):
        return self[self.argmax()]
    
    def min(self):
        return self[self.argmin()]
    
    def argmax(self):
        if len(self) == 0:
            raise Exception('Cannot take argmax/min without any choices')
        return max( [(value,key) for (key,value) in self.iteritems()] )[1]
    
    def argmin(self):
        if len(self) == 0:
            raise Exception('Cannot take argmax/min without any choices')
        return min( [(value,key) for (key,value) in self.iteritems()] )[1]
    
    def normalize(self):
        for val in self.values():
            if val < 0:
                raise Exception('Cannot normalize if numbers < 0')
        
        sumvals = sum(self.values())
        result = NumMap()
        for (key,val) in self.iteritems():
            result[key] = float(val)/sumvals
        return result
    
    def __getitem__(self,key):
        if not key in self:
            return 0.0
        return super(NumMap,self).__getitem__(key)
    
    def __str__(self):
        result_list = ["{"]
        for (k,v) in self.items():
            result_list.append( '{}:{:4.4f}, '.format(str(k), v) )
        result_list[-1] = result_list[-1][:-2]
        result_list.append('}')
        return ''.join(result_list)
    
    def __eq__(self, other):
        try:
            for (key,val) in self.items():
                if val != other[key]:
                    return False
            for (key, val) in other.items():
                if val != self[key]:
                    return False
            return True
        except Exception:
            return False
    
    def info(self):
        result = ['NumMap:\n']
        for (k,v) in self.items():
            result.append( '\t{} ===== {:4.4f}\n'.format(str(k), v))
        return ''.join(result)
    
class FDict(object):
    '''
    A dictionary-like class backed by a function.  Class is read-only
    and entries cannot be set or enumerated only.  Size is infinite.
    '''
    
    def eval(self, key):
        '''
        Equivalent of a key lookup.  Raise KeyError if this function
        isn't applicable for key's type.
        '''
        raise NotImplementedError()
    
    def __len__(self):
        return float('infinity')
    
    def __getitem__(self, key):
        return self.eval(key)
    
    def __contains__(self, key):
        try:
            self.eval(key)
            return True
        except KeyError:
            return False
        
class FeatureFunction(object):
    '''A feature function'''
    @property
    def dim(self):
        '''dimension of all output from self.features()'''
        raise NotImplementedError()
    
    def features(self, *args):
        '''Calculate features for arguments.  Returns
        numpy.ndarray of length self.dim'''
        raise NotImplementedError()

if __name__ == '__main__':
    a = NumMap()
    a['dave'] = 15
    a['steve'] = 30
    
    a = a.normalize()
    
    assert ( a['dave'] == 1.0/3 )
    assert ( a.argmax() == 'steve' )
    assert ( a.argmin() == 'dave'  )
    
    print('NumMap passes all tests')