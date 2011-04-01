
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

if __name__ == '__main__':
    a = NumMap()
    a['dave'] = 15
    a['steve'] = 30
    
    a = a.normalize()
    
    assert ( a['dave'] == 1.0/3 )
    assert ( a.argmax() == 'steve' )
    assert ( a.argmin() == 'dave'  )
    
    print('NumMap passes all tests')