import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_theme()

def plot_decision_boundary(model, X, y, mesh = 0.01):
    """ X must be size (len(X), 1, nb params = 2)
        Y must be size (len(Y), 1)"""
    
    x_min, x_max = X[:, :, 0].min() - 0.5, X[:,:,0].max() + 0.5
    y_min, y_max = X[:, :, 1].min() - 0.5, X[:,:, 1].max() + 0.5
    
    # meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh), np.arange(y_min, y_max, mesh))
    
    # predictions
    X_cont = np.c_[xx.ravel(), yy.ravel()].reshape(len(np.c_[xx.ravel(), yy.ravel()]), 1, 2)
    Z = np.array(model.predict(X_cont))
    Z = Z.reshape(Z.shape[0], Z.shape[2])
    
    res = np.array([np.argmax(i) for i in Z])
    res = res.reshape(xx.shape)
    
    X_1 = X.reshape(len(X),2)
    
    # Plot
    plt.contourf(xx, yy, res, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X_1[:, 0], X_1[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary Plot')
    plt.show()

def one_hot_encoder(y):
    shape = (y.size, int(np.max(y) + 1))
    rows = np.arange(y.size)
    one_hot = np.zeros(shape)
    one_hot[rows, y[:,0]] = 1.
    return one_hot