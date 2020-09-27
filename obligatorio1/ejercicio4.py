import numpy as np
import matplotlib.pyplot as plt
from obligatorio1.least_squares import LeastSquares
from obligatorio1.gradient_descent import GradientDescent

font = {'size': 16}
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


def save_error_log(error_log, filepath):
    plt.figure(figsize=(8,8))
    plt.plot(error_log)
    plt.grid()
    plt.title('Evolución del error relativo')
    plt.ylabel("$\\frac{||x^* - x^t||}{||x^*||}$")
    plt.xlabel('t')
    plt.savefig(filepath)


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
    iters = []

    # Part a) Fixed step
    optimizer = GradientDescent(least_squares)
    fixed_step = 0.5/np.linalg.norm(A, 2)**2
    x = optimizer.solve(max_iter, 'fixed', verbose=True, fixed_step=fixed_step)
    error_log = compute_relative_error(optimizer.x_log, x_ref)
    save_error_log(error_log, './images/ejercicio4_paso_fijo.png')
    iters.append(len(optimizer.x_log))

    # Part b) Decreasing step
    optimizer = GradientDescent(least_squares)
    x = optimizer.solve(max_iter, 'decreasing', verbose=True, base=0.001)
    error_log = compute_relative_error(optimizer.x_log, x_ref)
    save_error_log(error_log, './images/ejercicio4_decreciente.png')
    iters.append(len(optimizer.x_log))

    # Part c) Line search
    optimizer = GradientDescent(least_squares)
    x = optimizer.solve(max_iter, 'line_search', verbose=True, max_step=0.01, n_points=1000)
    error_log = compute_relative_error(optimizer.x_log, x_ref)
    save_error_log(error_log, './images/ejercicio4_line_search.png')
    iters.append(len(optimizer.x_log))

    # Part d) Armijo
    optimizer = GradientDescent(least_squares)
    x = optimizer.solve(max_iter, 'armijo', verbose=True, max_step=0.01, sigma=0.1, beta=0.5)
    error_log = compute_relative_error(optimizer.x_log, x_ref)
    save_error_log(error_log, './images/ejercicio4_armijo.png')
    iters.append(len(optimizer.x_log))

    # Bar plot iterations
    plt.figure(figsize=(12, 8))
    plt.bar(np.arange(len(iters)), height=iters)
    plt.title('Cantidad de iteraciones')
    plt.ylabel('t')
    plt.xticks(np.arange(len(iters)), labels=['Fijo', 'Decreiente', 'Line Search', 'Armijo'])
    plt.savefig('./images/ejercicio4_iters.png')

