from Layer import Layer

class Activation(Layer):
    """Base class of an activation layer"""
    def __init__(self, name, activation_function, activation_deriv):
        self.name = name
        self.activation_function = activation_function
        self.activation_deriv = activation_deriv
        
    def forward_propagation(self, x_input):
        self.input = x_input
        self.output = self.activation_function(self.input)
        return self.output
    
    def backward_propagation(self, output_gradient, learning_rate):
        return output_gradient * self.activation_deriv(self.input)