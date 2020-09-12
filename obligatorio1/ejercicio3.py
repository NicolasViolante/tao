import numpy as np
import matplotlib.pyplot as plt
from least_squares import LeastSquares
from gradient_descent import GradientDescent
            
def compute_relative_error(x_log, x):
    """
    Returns relative error between final value x and previous predictions 
    stored in x_log 
    
    Inputs:
        x : Numpy array of shape (N,)   
        x_log : Numpy array of shape (iterations, N)
    """
    return np.linalg.norm(x_log -x, axis=1)/np.linalg.norm(x)


def solve_normal_equations(A,b):
    """
    Returns exact solution to least squares problem
    """
    x = np.matmul(np.linalg.pinv(np.matmul(np.transpose(A), A)), np.matmul(np.transpose(A),b))
    return x


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
    max_iter = 500
    step_schedules = ['decreasing']
    
    error_logs = []
    for step in step_schedules:
        least_squares = LeastSquares(A, b)
        optimizer = GradientDescent(least_squares)
        x = optimizer.solve(max_iter, step=step, verbose=True)
        
        # error_log = compute_relative_error(least_squares.x_log, x_ref)
        # error_logs.append(error_log)
    
    # Plot error evolution
    #------------------------------------------------------------
    # plt.figure()
    # for error_log, step_schedule in zip(error_logs, step_schedules):
    #     plt.plot(error_log, label=step_schedule)
    # plt.grid()
    # plt.ylabel('Error relativo')
    # plt.xlabel('Iteraciones')
    # plt.legend()
    # plt.savefig('./images/ejercicio3.pdf')