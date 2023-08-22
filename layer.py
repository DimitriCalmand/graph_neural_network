import numpy_graph as npg
import numpy as np

def randome(shape):
    val = 1
    for i in range (len(shape)):
        val *= shape[i]
    l = []
    for i in range(val):
        l.append(i/100)
    return np.array(l).reshape(shape)
class layer:
    def __init__(self):
        self.input = None
        self.output = None
    def call(self, input : np.ndarray):
        pass
    def build(self,input):
        pass
    def add_trainable_array(self, shape, is_bias = False):
        array = np.random.randn(*shape)/3 
        #array = randome(shape)
        #if is_bias:
        #    array = np.zeros(shape)
        res = npg.g_array(array, trainable = True, is_bias = is_bias)
        return res
