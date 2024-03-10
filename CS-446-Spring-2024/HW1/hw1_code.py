import numpy as np
import hw1_utils as utils
from numpy.linalg import inv
import matplotlib.pyplot as plt

def linear_gd(X,Y,lrate=0.1,num_iter=1000): 
    Y = Y[:,np.newaxis] # resize Y to col vector
    X  = np.hstack((np.ones(((X.shape[0]),1)),X)) # add columns of 1 to front of X
    X=X.T # X is tranpose of itself
    w = np.zeros((X.shape[0],1)) # initialize w in R^(d+1)
    for _ in range(num_iter): 
        grad = ((X@X.T)@w) - (X@Y) # calc grad at w
        grad = grad * lrate * (1/Y.shape[0]) # multiplity with learning rate and constant
        w = w - grad     # update w
    return w

def linear_normal(X,Y):
    Y = Y[:,np.newaxis] #same steps as in linear_gd function
    X  = np.hstack((np.ones(((X.shape[0]),1)),X))
    X=X.T
    w = np.zeros((X.shape[0],1))
    w = ((inv(X@X.T))@X)@Y # compute analytically using given formula
    return w

def plot_linear():
    X,Y = utils.load_reg_data()
    w = linear_normal(X,Y)
    X  = np.hstack((np.ones(((X.shape[0]),1)),X))
    y_predict = X@w # create predicted Y
    y_predict = y_predict.reshape(-1) # reshape it Y
    plt.title("Trained vs Predicted Dataset")
    plt.scatter(X[:, 1], Y, label='Y_train', color='blue', marker='x')
    plt.plot(X[:, 1], y_predict, label='Y_predict', color='red')  # color code the plots
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show() # show graph
    return []
# plot_linear()