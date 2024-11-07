from Layer import Layer
import numpy as np


class FlattenLayer():
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.name = "Flatten"
        self.input_size = input_shape
        self.output_size = "(1, -1)"
        
    def forward_propagation(self, x_input):
        self.output = np.reshape(x_input, (1, -1))
        return self.output
    
    def backward_propagation(self, output_error, learning_rate):
        return np.reshape(output_error, self.input_shape)

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        
    def forward_propagation(self, x_input):
        return np.reshape(x_input, self.output_shape)
    
    def backward_propagation(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)
    
