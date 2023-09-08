import numpy as np 
import matplotlib.pyplot as plt
import math

def inv_jacobian(z):
    x = z[0]
    y = z[1]
    j = np.array([[6*x*y, 3*x*x - 3*y*y],[3*x*x - 3*y*y, -6*x*y]])
    return np.linalg.inv(j)

def derivate(z):
    x = z[0]
    y = z[1]
    der = np.array([3*x*x*y - y**3, x**3-3*x*y*y-1])
    return der

def newton_method(z0: np.array, N=50)->np.array: 
    z = z0
    for i in range(N):
        z = z - inv_jacobian(z) @ derivate(z)
    return z

def plot_image(s_points: np.array, n=500, domain=(-1, 1,-1, 1)): 
    m = np.zeros((n, n)) 
    xmin, xmax, ymin, ymax = domain 
    for ix, x in enumerate(np.linspace(xmin, xmax, n)): 
        for iy, y in enumerate(np.linspace(ymin, ymax, n)): 
            z0 = np.array([x,y]) 
            zN = newton_method(z0)
            code = np.argmin(np.linalg.norm(s_points-zN,ord=2,axis=1)) 
            m[iy, ix] = code 
    
    plt.imshow(m, cmap="brg") 
    plt.axis("off") 
    plt.savefig("q2_hw3.png") 
                                                                
if __name__ == "__main__": 
    stationary_points= np.array([[-1,0],[1,0],[-0.5,0.5*math.sqrt(3)],[-0.5,-0.5*math.sqrt(3)]]) 
    plot_image(stationary_points)