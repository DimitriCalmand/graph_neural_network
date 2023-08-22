import numpy_graph as npg
import numpy as np
import layer

class Dense(layer.layer):
    def __init__(self, output_size , input_shape =  None):
        self.output_size = output_size 
        self.input_shape = input_shape
    def build(self, input):
        self.input_shape = input.shape
        self.w = self.add_trainable_array(
                (self.input_shape[1], self.output_size)
                )
        self.b = self.add_trainable_array(
                (1, self.output_size), is_bias = True
                )
    def call(self, input):
        return input @ self.w + self.b

