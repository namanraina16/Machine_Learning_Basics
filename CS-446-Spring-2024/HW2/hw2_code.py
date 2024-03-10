import hw2_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw2_utils.poly(degree=1), c=None):
    alpha = torch.zeros(x_train.shape[0], requires_grad=True)
    for _ in range(num_iters):
        new = alpha - lr * compute_gradient(alpha, x_train, y_train, kernel)
        alpha_new = clamped_projection(c, new)
        alpha = alpha_new
    return alpha.detach()
def compute_gradient(alpha, x_train, y_train, kernel):
    K_test_train = torch.zeros(x_train.shape[0],x_train.shape[0])
    for i in range(x_train.shape[0]):
        for j in range(x_train.shape[0]):
            K_test_train[i][j]=kernel(x_train[i],x_train[j])
    y_outer = torch.outer(y_train, y_train)
    grad_first_term = torch.matmul(y_outer * K_test_train, alpha)
    grad_second_term = torch.sum(alpha)
    gradient = grad_first_term - 1
    return gradient
def clamped_projection(C, alpha):
    projected_alpha = torch.clamp(alpha, min=0, max=C)
    return projected_alpha     
def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw2_utils.poly(degree=1)):
    K_test_train = torch.zeros(x_test.shape[0],x_train.shape[0])
    for i in range(x_test.shape[0]):
        for j in range(x_train.shape[0]):
            K_test_train[i][j]=kernel(x_test[i],x_train[j])
    preds = torch.matmul(K_test_train, alpha * y_train)
    return preds
def plot_svm_contour(x_xor, y_xor, kernels, kernel_names, lr=0.1, num_iters=10000):
    for kernel in kernels:
        alphas = svm_solver(x_xor, y_xor, lr, num_iters, kernel)
        hw2_utils.svm_contour(lambda x: svm_predictor(alphas, x_xor, y_xor, x, kernel))
x_xor, y_xor = hw2_utils.xor_data()
kernels = [hw2_utils.poly(degree=2), 
           hw2_utils.rbf(sigma=1), 
           hw2_utils.rbf(sigma=2), 
           hw2_utils.rbf(sigma=4)]
kernel_names = ['Poly (Degree=2)', 'RBF (Sigma=1)', 'RBF (Sigma=2)', 'RBF (Sigma=4)']
#plot_svm_contour(x_xor, y_xor, kernels, kernel_names)