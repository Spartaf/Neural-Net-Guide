import numpy as np
from Activation import Activation

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_deriv = lambda x: 1 - np.tanh(x) ** 2
        super().__init__("Tanh", activation_function = tanh, activation_deriv = tanh_deriv)
    

class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        sigmoid_deriv = lambda x: np.exp(-x) / (1 + np.exp(-x))**2
        super().__init__("Sigmoid", activation_function = sigmoid, activation_deriv = sigmoid_deriv)


class Relu(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(x, 0)
        relu_deriv = lambda x: np.array(x > 0).astype('int')
        super().__init__("Relu", activation_function = relu, activation_deriv = relu_deriv)



class Softmax(Activation):
    def __init__(self, input_size):
        self.name = "Softmax"
        self.input_size = input_size
        self.output_size = input_size
        
    def forward_propagation(self, x_input):
       # print(x_input.shape)
        tmp = np.exp(x_input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward_propagation(self, output_gradient, learning_rate):
        n = np.size(self.output)
        M = np.tile(self.output, (n,1))
#         print("output shape : " ,self.output.shape)
#         print("M shape : ", M.shape)
#         print("output gradient shape : ",output_gradient.shape)
#         print((M.T * (np.eye(n) - M)).shape)
        
        return output_gradient @ (M.T * (np.eye(n) - M))