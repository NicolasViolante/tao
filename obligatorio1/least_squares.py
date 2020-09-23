import numpy as np


class LeastSquares(object):
    """
    Inputs:
    - A : np array of shape (M,N)
    - b : np array of shape (M,)
    """
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.input_size = A.shape[1]
        
    def forward(self, x):
        """
        Computes f(x)
        """
        u = np.matmul(self.A, x) - self.b      # shape (M,)
        f = np.matmul(np.transpose(u), u)      # scalar
        return f
    
    def backward(self, x):
        """
        Computes df/dx
        """
        u = np.matmul(self.A, x) - self.b      # shape (M,)
        grad_u = 2*u 
        grad_x = np.matmul(grad_u, self.A)     # shape (N,)
        return grad_x 
