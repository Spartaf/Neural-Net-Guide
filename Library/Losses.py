import numpy as np 

class MSE():
    def compute_loss(self, y_true , y_pred):
        return np.mean(np.power(y_true - y_pred, 2))
    def compute_loss_grad(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)

class binary_cross_entropy():
    def compute_loss(self, y_true , y_pred):
        return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
    def compute_loss_grad(self, y_true, y_pred):
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)


class CategoricalCrossentropy():
    def compute_loss(self, y_true, y_pred):
        # Clip values to avoid log(0) issues
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        # Calculate categorical crossentropy loss
        loss = -np.sum(y_true * np.log(y_pred)) / len(y_true)
        return loss

    def compute_loss_grad(self, y_true, y_pred):
        # Clip values to avoid division by zero issues
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        # Calculate categorical crossentropy gradient
        grad = -y_true / y_pred / len(y_true)
        return grad