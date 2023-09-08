import asgn_source as asou 
import numpy as np 
import scipy
from scipy.optimize import minimize

def project_coordinate(x,constraint_min,constraint_max): 
    x = np.where(x > constraint_max, constraint_max, x)
    x_projected = np.where(x < constraint_min, constraint_min, x)
    return x_projected


def run_projected_GD(constraint_min,constraint_max,Q,b,c,n,step,epsilon,beta,sigma):
    x = np.random.rand(n)
    coef = [Q, b, c]
    result = pgd_armij(coef, x, step, epsilon, beta, sigma, constraint_min,constraint_max)
    return result


def gradient(x, coef):
    return coef[0] @ x + coef[1]


def pgd_armij(coef, x_start, step, epsilon, beta, sigma, constraint_min,constraint_max):
    iter = 0
    x_i = x_start
    while True:
        x_j = armij_rl(coef, x_i, step, beta, sigma)
        x_j = project_coordinate(x_j, constraint_min,constraint_max)
        iter += 1
        if np.linalg.norm(x_i - x_j) < epsilon:
            break
        x_i = x_j
    return x_j, iter


def armij_rl(coef, x, step, beta, sigma):
    m_k = 0
    f_x = quad_func(x, coef)
    grad_f = gradient(x, coef)
    diff = np.linalg.norm(grad_f) ** 2 * sigma * step
    while quad_func(x - grad_f * step * beta ** m_k, coef) > f_x - diff:
        m_k += 1
        diff *= beta
    return x - grad_f * step * beta ** m_k


def quad_func(x, coef):
    x = np.array(x)
    return 0.5 * x.T @ coef[0] @ x + coef[1].T @ x + coef[2]

def const_upper(x, constraint_max):
    return constraint_max - x

def const_lower(x, constraint_min):
    return x - constraint_min

def verify_result(n, Q_val, b_val, c_val, constraint_max, constraint_min):
    con1 = {'type': 'ineq', 'fun': const_upper, 'args': ([constraint_max])}
    con2 = {'type': 'ineq', 'fun': const_lower, 'args': ([constraint_min])} 
    cons = [con1, con2]
    solution = minimize(quad_func,x0=np.zeros(n), args=([Q_val,b_val,c_val]), method='SLSQP',bounds=None, constraints=cons)
    return solution.x

def main():
    step = 0.1
    epsilon = 0.0001
    beta = 0.3
    sigma = 1e-2
    n =25 
    np.random.seed() 
    constraint_min,constraint_max, Q_val,b_val, c_val = asou.get_parameters(n)
    armijo_sol = run_projected_GD(constraint_min,constraint_max,Q_val,b_val, c_val,n,step,epsilon,beta,sigma)
    print(armijo_sol[0])
    x = verify_result(n, Q_val, b_val, c_val, constraint_max, constraint_min)
    print(x)

if __name__ == '__main__':
    main()