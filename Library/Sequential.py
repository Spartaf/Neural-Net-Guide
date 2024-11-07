import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from sklearn.metrics import accuracy_score
from Activation import Activation
from Layers import FlattenLayer


from Losses import *

class Sequential():
    """Class of a Sequential network"""
    def __init__(self, layers = []):
        self.layers = layers
        self.history = {'Loss' : [], 'Acc' : []}
    
    def add(self, layer):
        self.layers.append(layer)

    def add_model(self , model):
        # Tranfere aussi les poids
        for layer in model.layers:
            self.layers.append(layer) 
            
    def summary(self):
        nb_params = 0
        for layer in self.layers:
            if isinstance(layer , Activation) or layer.__class__ == FlattenLayer:
                print(f'{layer.name}()')
            else:
                print(f'{layer.name}() : input shape = {layer.input_size}, output_shape = {layer.output_size}')
                nb_params += layer.weights.shape[0] * layer.weights.shape[1] # ajout du nombre de poids
                if layer.use_bias:
                    nb_params += layer.bias.shape[0] * layer.bias.shape[1]   
        print(f'Le model a {nb_params} paramètres')
                
    def get_weights(self):
        w_dict = {}
        temp = np.arange(0, len(self.layers))
        for layer, i in zip(self.layers, temp):
            if not isinstance(layer , Activation) and not layer.__class__ == FlattenLayer :
                if layer.use_bias:
                    w_dict[f"Layer {i}-{layer.name}"] = {"Weights" : layer.weights, "Bias" : layer.bias}
                else:
                    w_dict[f"Layer {i}-{layer.name}"] = {"Weights" : layer.weights}
        return w_dict
    
    def predict(self,  x_input):
        samples = len(x_input)
        result = []
        for i in range(samples):
            output = x_input[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result
    
    def compiler(self, loss = MSE(), learning_rate = 0.01):
        self.loss = loss
        self.learning_rate = learning_rate
        
    def fit(self, x_train, y_train, epochs = 10, learning_rate = 0.01 ,verbose = True):
        """ x_train must be size : len(x_train), 1, nb_params 
            y_train must be size : len(y_train),  nb_params
            y_train must be encode in one hot"""
        samples = len(x_train)
        
        for epoch in range(epochs):
            error = 0
            y_preds = []
            y_true = []
            for x, y in zip(x_train, y_train):
                
                output = x
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                
                y_preds.append(np.argmax(output))
                y_true.append(np.argmax(y))
                error += self.loss.compute_loss(y, output)
                
                output_gradient = self.loss.compute_loss_grad(y, output)
                
                for layer in reversed(self.layers):
                    output_gradient = layer.backward_propagation(output_gradient, self.learning_rate)
            
            error = error / samples
            acc = accuracy_score(y_true, y_preds)
            self.history['Acc'].append(acc)
            self.history['Loss'].append(error)
            if verbose:
                print(f"{epoch + 1}/{epochs}, error={error :.4f}, acc={acc}")
            
                
    def plot_history_acc(self):
        fig, ax = plt.subplots(1, 2, figsize= (10, 3))
        ax[0].plot(self.history['Loss'])
        ax[0].set_title("Loss")
        ax[1].plot(self.history['Acc'])
        ax[1].set_title("Accuracy")
        plt.show()