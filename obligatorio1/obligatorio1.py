import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(A, b, step, iterations):
    """
    Optimize ||Ax -b||2 with gradient descent
    Inputs:
        A : Numpy array of shape (M,N)
        b : Numpy array of shape (M,)
        step : String specifing type of step.
        iterations: Int, number of iterations before stopping
    """

    def relative_error(x_log, x):
        """
        Returns relative error between final value x and previous predictions 
        stored in x_log 
        Inputs:
            x : Numpy array of shape (N,)   
            x_log : Numpy array of shape (iterations, N)
        """
        return np.linalg.norm(x_log -x, axis=1)/np.linalg.norm(x)

    def initialize(size):
        return np.random.randn(size)

    def fixed_step(*args):
        A = args[0]
        return 0.5/np.linalg.norm(A, ord=2)**2
    


    def line_search():
        pass
    
    def decreasing_step(*args, base=0.01):
        k = args[2]
        return base / (k+1)

    def arminjo():
        pass
    
    def learning_rate(step_fn, *args):
        if step_fn == 'arminjo': 
            return arminjo(*args)
        elif step_fn == 'line search':
            return line_search(*args)
        elif step_fn == 'decreciente':
            return decreasing_step(*args)
        elif step_fn == 'fijo':
            return fixed_step(*args)
       

    M, N = A.shape
    x_log = np.empty((iterations, N)) 

    x = initialize(N)

    for k in range(iterations):
        x_log[k] = x

        # Forward pass
        u = np.matmul(A,x) - b             # shape (M,1)
        f = np.matmul(np.transpose(u), u)  # scalar
         
        # print('[{}/{}] f {}'.format(k,iterations, f))

        # Backward pass
        grad_u = 2*np.transpose(u)         # shape (1,M)
        grad_x = np.matmul(grad_u, A)      # shape (1,N), but (N,) for Numpy 

        # Update
        x = x - learning_rate(step, A,b,k)*grad_x 

    error = relative_error(x_log, x)

    return x, error

def normal_equations(A,b):
    return np.matmul(np.linalg.pinv(np.matmul(np.transpose(A), A)), np.matmul(np.transpose(A),b))


if __name__ == '__main__':
    # Load data
    #---------------------------------------------------------
    A = np.loadtxt('./archivos_ob1/A.asc')
    b = np.loadtxt('./archivos_ob1/b.asc')

    # Run gradient descent
    #---------------------------------------------------------
    n_iterations = 1000
    steps = ['decreciente']#,'line search', 'arminjo']
    xs = []
    errors = []
    for step in steps:
        x, error = gradient_descent(A, b, step, n_iterations)
        xs.append(x)
        errors.append(error) 

    # Show results
    # #------------------------------------------------------- 
    # plt.figure()
    # for error, step in zip(errors, steps):
    #     plt.plot(error, label=step)

    # plt.grid()
    # plt.legend()
    # plt.show()
    
    for error, step in zip(errors, steps):
        plt.figure()
        plt.plot(error, label=step)
        plt.show()

    print()
    
        
    