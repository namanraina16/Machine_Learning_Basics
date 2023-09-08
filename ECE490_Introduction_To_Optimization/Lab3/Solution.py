import numpy as np 
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def get_Q_A_b(m,n): 
    Q = np.random.rand(n,n)-0.5 
    Q = 10*Q @ Q.T + 0.1*np.eye(n) 
    A = np.random.normal(size=(m,n)) 
    b = 2*(np.random.rand(m)-0.5) 
    return Q,A,b

def method_of_multiplier(Q, A, b, lamb_init, c_init, x_init, step, epsilon, beta, sigma):
    iter = 0
    x = []
    x_i = x_init
    lamb_i = lamb_init
    c_i = c_init
    while True:
        x.append(x_i)
        x_j = GD_armij(Q, A, b, lamb_i, c_i, x_i, step, epsilon, beta, sigma)
        lamb_j = lamb_i + c_i * (A @ x_j - b)
        c_j = c_sequence_a(c_i, x_i, x_j, A, b)
        iter += 1
        if np.linalg.norm(A @ x_j - b) < epsilon:
            break
        x_i = x_j
        lamb_i = lamb_j
        c_i = c_j
        print("Iteration: "+str(iter))
    return x_j, iter, x

def GD_armij(Q, A, b, lamb_i, c_i, x_i, step, epsilon, beta, sigma):
    x_i = x_i
    while True:
        x_j = armij_rl(Q, A, b, lamb_i, c_i, x_i, step, beta, sigma)
        if np.linalg.norm(aug_grad(Q, A, b, lamb_i, c_i, x_j)) < epsilon:
            break
        x_i = x_j
    return x_i

def armij_rl(Q, A, b, lamb, c, x, step, beta, sigma):
    m_k = 0
    f_x = aug_lagrangian(Q, A, b, lamb, c, x)
    grad_f = aug_grad(Q, A, b, lamb, c, x)
    diff = np.linalg.norm(grad_f) ** 2 * sigma * step
    while aug_lagrangian(Q, A, b, lamb, c, x - grad_f * step * beta ** m_k) > f_x - diff:
        m_k += 1
        diff *= beta
    return x - grad_f * step * beta ** m_k


def aug_lagrangian(Q, A, b, lamb, c, x):
    x = np.array(x)
    return x.T @ Q @ x + lamb.T @ (A @ x - b) + c * np.linalg.norm(A @ x - b) ** 2

def quad_func(x, Q):
    x = np.array(x)
    Q = np.array(Q)
    return x.T @ Q @ x


def aug_grad(Q, A, b, lamb, c, x):
    return 2 * Q @ x + A.T @ lamb.T + c * 2 * (A.T @ (A @ x - b))


def verify_result(n, Q, A, b):
    x0 = np.zeros(n)
    cons = [{'type': 'eq', 'fun': lambda x: A @ x - b}]
    solution = minimize(quad_func, x0=x0, args=([Q]), method='SLSQP',bounds=None, constraints=cons)
    return solution.x

def c_sequence_a(c_i, x_i, x_j, A, b):
    return c_i

def c_sequence_b(c_i, x_i, x_j, A, b):
    beta = 10
    return c_i + beta

def c_sequence_c(c_i, x_i, x_j, A, b):
    beta = 1.2
    return c_i * beta

def c_sequence_d(c_i, x_i, x_j, A, b):
    beta = 1.1
    gamma = 0.25
    if np.linalg.norm(A @ x_j - b) > gamma * np.linalg.norm(A @ x_i - b):
        c_j = c_i * beta
    else:
        c_j = c_i
    return c_j

def main():
    np.random.seed(100)

    step = 0.1
    epsilon = 1e-4
    beta = 0.3
    sigma = 1e-2
    m = 10
    n = 25
    lamb_init = np.random.rand(m)
    c_init = 2
    x_init = np.random.rand(n)


    Q, A, b = get_Q_A_b(m, n)
    armijo_sol = method_of_multiplier(Q, A, b, lamb_init, c_init, x_init, step, epsilon, beta, sigma)
    print("minimizer: "+str(armijo_sol[0]))
    print("min value: "+str(quad_func(armijo_sol[0], Q)))
    print("iteration: "+str(armijo_sol[1]))
    x = verify_result(n, Q, A, b)
    print("minimizer(ideal): "+str(x))
    print("min value(ideal): "+str(quad_func(x, Q)))


    iter = np.arange(armijo_sol[1])
    fig, ax = plt.subplots()
    x_k = np.array(armijo_sol[2])
    x = np.array(x)
    norm_error = np.zeros(armijo_sol[1])
    for i in range(armijo_sol[1]):
        norm_error[i] = np.linalg.norm(x_k[i]-x)/np.linalg.norm(x)
    ax.plot(iter, norm_error, '-o', c='black', linewidth=2)
    plt.xlabel('Iteration', fontsize=15)
    plt.ylabel('Normialized Error', fontsize=15)
    plt.show()


if __name__ == '__main__':
    main()