import numpy as np
import matplotlib.pyplot as plt
from obligatorio1.least_squares import LeastSquares
from obligatorio1.gradient_descent import GradientDescent


def compute_relative_error(x_log, x):
    """
    Returns relative error between final value x and previous predictions 
    stored in x_log 
    
    Inputs:
        x : Numpy array of shape (N,)   
        x_log : Numpy array of shape (iterations, N)
    """
    return np.linalg.norm(x_log -x, axis=1)/np.linalg.norm(x)


def solve_normal_equations(A, b):
    """
    Returns exact solution to least squares problem
    """
    x = np.matmul(np.linalg.pinv(np.matmul(np.transpose(A), A)), np.matmul(np.transpose(A),b))
    return x


def save_error_log(error_log, step_schedule):
    plt.figure(figsize=(8,8))
    plt.plot(error_log, label=step_schedule)
    plt.grid()
    plt.ylabel("$\\frac{||x^* - x^t||}{||x^*||}$")
    plt.xlabel('t')
    plt.legend()
    plt.savefig('./images/ejercicio3_{}.pdf'.format(step_schedule))


if __name__ == '__main__':
    # Load data
    #------------------------------------------------------------
    A = np.loadtxt('./archivos_ob1/A.asc')
    b = np.loadtxt('./archivos_ob1/b.asc')

    # Reference solution
    #------------------------------------------------------------
    x_ref = solve_normal_equations(A,b)
    print('Reference solution is x = {}'.format(x_ref))
    
    # Least Squares minimization
    #------------------------------------------------------------
    max_iter = 2000
    least_squares = LeastSquares(A, b)

    # Part a) Fixed step
    optimizer = GradientDescent(least_squares)
    x = optimizer.solve(max_iter, step=0.5/np.linalg.norm(A)**2, verbose=True)
    error_log = compute_relative_error(optimizer.x_log, x_ref)
    save_error_log(error_log, 'Paso fijo')

    # Part b) Decreasing step
    optimizer = GradientDescent(least_squares)
    x = optimizer.solve(max_iter, step='decreasing', verbose=True)
    error_log = compute_relative_error(optimizer.x_log, x_ref)
    save_error_log(error_log, 'Paso decreciente')

    # Part c) Line search

    # Part d) Arminjo

