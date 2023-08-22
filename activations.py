import numpy as np
import numpy_graph as npg
import layer

class sigmoid(layer.layer) :
    def __init__ (self) -> None:        
        pass
    def build(self, input):
        pass
    def call(self, input):
        res = 1/(1+npg.exp(-input))
        return res

