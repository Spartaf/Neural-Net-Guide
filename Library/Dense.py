import numpy as np
from Layer import Layer

class Dense(Layer):
    """Class of the Dense layer"""
    def __init__(self, input_size, output_size, use_bias = True):
        self.name = "Dense"
        self.use_bias = use_bias
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size + output_size)
        if use_bias:
            self.bias = np.random.randn(1, output_size) / np.sqrt(input_size + output_size) 
    
    def forward_propagation(self, x_input):
        self.input = x_input
        if self.use_bias:
            self.output = np.dot(self.input, self.weights) + self.bias
        else:
            self.output = np.dot(self.input, self.weights)
        return self.output
    
    def backward_propagation(self, output_gradient, learning_rate):
        input_gradient = np.dot(output_gradient, self.weights.T)
        self.weights_gradient = np.dot(self.input.T, output_gradient)
        
        if self.use_bias:
            self.bias_gradient = output_gradient
            self.bias = self.bias - learning_rate * self.bias_gradient
        self.weights = self.weights - learning_rate * self.weights_gradient
        
        return input_gradient