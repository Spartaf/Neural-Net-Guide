
class Layer():
    """ Base Class of a simple layer"""
    def __init__(self):
        self.name = None
        self.input = None
        self.output = None
    
    def forward_propagation(self, x_input):
        pass
    def backward_propagation(self, output_gradient, learning_rate):
        pass