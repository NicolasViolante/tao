import numpy as np
import matplotlib.pyplot as plt
from obligatorio1.least_squares import LeastSquares
from obligatorio1.gradient_descent import GradientDescent

font = {'family': 'normal',
        'size': 16}
plt.rc('font', **font)


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
    plt.plot(error_log)
    plt.grid()
    plt.ylabel("$\\frac{||x^* - x^t||}{||x^*||}$")
    plt.xlabel('t')
    plt.title(step_schedule)
    plt.savefig('./images/ejercicio4_{}.png'.format(step_schedule))


if __name__ == '__main__':
    # Load data
    #-------------------------------------------------------------------------------------------------------------------
    A = np.loadtxt('./archivos_ob1/A.asc')
    b = np.loadtxt('./archivos_ob1/b.asc')

    # Reference solution
    #-------------------------------------------------------------------------------------------------------------------
    x_ref = solve_normal_equations(A,b)
    print('Reference solution is x = {}'.format(x_ref))
    
    # Least Squares minimization
    #-------------------------------------------------------------------------------------------------------------------
    max_iter = 1000
    least_squares = LeastSquares(A, b)

    # Part a) Fixed step
    optimizer = GradientDescent(least_squares)
    fixed_step = 0.5/np.linalg.norm(A)**2
    x = optimizer.solve(max_iter, 'fixed', verbose=True, fixed_step=fixed_step)
    error_log = compute_relative_error(optimizer.x_log, x_ref)
    save_error_log(error_log, 'Paso fijo')

    # # Part b) Decreasing step
    optimizer = GradientDescent(least_squares)
    x = optimizer.solve(max_iter, 'decreasing', verbose=True, base=0.001)
    error_log = compute_relative_error(optimizer.x_log, x_ref)
    save_error_log(error_log, 'Paso decreciente')

    # Part c) Line search
    optimizer = GradientDescent(least_squares)
    x = optimizer.solve(max_iter, 'line_search', verbose=True, max_step=1e-4, n_points=100)
    error_log = compute_relative_error(optimizer.x_log, x_ref)
    save_error_log(error_log, 'Line Search')

    # Part d) Armijo
    optimizer = GradientDescent(least_squares)
    x = optimizer.solve(max_iter, 'armijo', verbose=True, max_step=1e-4, sigma=0.1, beta=0.5)
    error_log = compute_relative_error(optimizer.x_log, x_ref)
    save_error_log(error_log, 'Armijo')

