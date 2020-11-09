import numpy as np
from numpy.linalg import inv, pinv, norm
import matplotlib.pyplot as plt

def compute_relative_error(x_log, x):
    """
    Returns relative error between final value x and previous predictions 
    stored in x_log 
    
    Inputs:
        x : Numpy array of shape (N,)   
        x_log : Numpy array of shape (iterations, N)
    """
    return norm(x_log -x, axis=1)/norm(x)

if __name__ == '__main__':
    # Load data
    #------------------------------------------------------------------------------
    A = np.loadtxt('./datos/A.txt')
    B = np.loadtxt('./datos/B.txt')
    C = np.loadtxt('./datos/C.txt')
    alpha = np.loadtxt('./datos/alpha.txt')
    beta = np.loadtxt('./datos/beta.txt')
    gamma = np.loadtxt('./datos/gamma.txt')
    
    n = A.shape[1]
    print(f'Dimension of x is n={n}')
    
    # Create auxiliary matrices
    #------------------------------------------------------------------------------
    O = np.zeros_like(A)
    I = np.eye(n)
    D = np.hstack([np.vstack([A, O, O]), np.vstack([O, B, O]), np.vstack([O, O, C])])
    H = np.vstack([np.hstack([I, -I, O]), np.hstack([O, I, -I])])
    delta = np.concatenate((alpha, beta, gamma))
    
    # Reference solution
    #------------------------------------------------------------------------------
    x_ref = inv(A.T@A + B.T@B + C.T@C)@(A.T@alpha + B.T@beta + C.T@gamma)    
    print(f'Reference solution if x={x_ref}')
    lambda_ref = inv(H@inv(D.T@D)@H.T)@H@inv(D)@delta
    w_ref = inv(D)@delta - inv(D.T@D)@H.T@lambda_ref
    # (w_ref[:10]+w_ref[10:20]+w_ref[20:30])/3
    
    iterations = 100
    # Método de penalización cuadrática
    # Parte e) i)
    #------------------------------------------------------------------------------
    L = lambda_ref
    x_log_pen_1 = []
    for k in range(iterations):
        if k>0:
            w_old = w
        tau = 2**k / norm(D, 2)
        w = inv(D.T@D + tau*H.T@H)@(D.T@delta - H.T@L)
        x = (w[:10]+w[10:20]+w[20:30])/3
        x_log_pen_1.append(x)
        if k>0 and norm(w-w_old)/norm(w) < 1e-8:
            break

    # Método de penalización cuadrática
    # Parte e) ii)
    #------------------------------------------------------------------------------
    L = np.zeros_like(lambda_ref)
    x_log_pen_2 = []
    for k in range(iterations):
        if k>0:
            w_old = w
        tau = 2**k / norm(D, 2)
        w = inv(D.T@D + tau*H.T@H)@(D.T@delta - H.T@L)
        x = (w[:10]+w[10:20]+w[20:30])/3
        x_log_pen_2.append(x)
        
        if k>0 and norm(w-w_old)/norm(w) < 1e-8:
            break
    
    # Método de los multiplicadores
    # Parte f) i)
    #------------------------------------------------------------------------------
    L = np.zeros_like(lambda_ref)
    x_log_mult_1 = []   
    tau = 10/norm(D, 2)   
    for k in range(iterations):
        if k>0:
            w_old = w
        L = L + tau*H@w
        w = inv(D.T@D + tau*H.T@H)@(D.T@delta - H.T@L)
        x = (w[:10]+w[10:20]+w[20:30])/3
        x_log_mult_1.append(x)
        
        if k>0 and norm(w-w_old)/norm(w) < 1e-8:
            break
        
    
    # Método de los multiplicadores
    # Parte f) ii)
    #------------------------------------------------------------------------------
    L = np.zeros_like(lambda_ref)
    x_log_mult_2 = []
    tau = 1000/norm(D, 2)
    for k in range(iterations):
        if k>0:
            w_old = w
        L = L + tau*H@w
        w = inv(D.T@D + tau*H.T@H)@(D.T@delta - H.T@L)
        x = (w[:10]+w[10:20]+w[20:30])/3
        x_log_mult_2.append(x)
        
        if k>0 and norm(w-w_old)/norm(w) < 1e-8:
            break
        
    # Método de los multiplicadores
    # Parte g)
    #------------------------------------------------------------------------------
    L = np.zeros_like(lambda_ref)
    x_log_comb_1 = []
    tau = 1000/norm(D, 2)
    for k in range(iterations):
        if k>0:
            w_old = w
        tau = 2**k / norm(D, 2)
        L = L + tau*H@w
        w = inv(D.T@D + tau*H.T@H)@(D.T@delta - H.T@L)
        x = (w[:10]+w[10:20]+w[20:30])/3
        x_log_comb_1.append(x)
        
        if k>0 and norm(w-w_old)/norm(w) < 1e-8:
            break
        
    # Plot resultados
    #------------------------------------------------------------------------------
    
    error_log_pen_1 = compute_relative_error(x_log_pen_1, x_ref)
    error_log_pen_2 = compute_relative_error(x_log_pen_2, x_ref)
    error_log_mult_1 = compute_relative_error(x_log_mult_1, x_ref)
    error_log_mult_2 = compute_relative_error(x_log_mult_2, x_ref)
    error_log_comb_1 = compute_relative_error(x_log_comb_1, x_ref)

    
    plt.figure(figsize=(8,6))
    plt.semilogy(error_log_pen_1, label='$\\lambda^k = \\lambda^*$, $\\tau^k = \\tau_02^k$')
    plt.semilogy(error_log_pen_2, label='$\\lambda^k = 0$, $\\tau^k = \\tau_02^k$')
    plt.semilogy(error_log_mult_1, label='$\\lambda^k = \\lambda^{k-1} + \\tau^k Hw$, $\\lambda^0 = 0$, $\\tau^k = 10\\tau_0$')
    plt.plot(error_log_mult_2, label='$\\lambda^k = \\lambda^{k-1} + \\tau^k Hw$, $\\lambda^0 = 0$, $\\tau^k = 1000\\tau_0$')
    plt.semilogy(error_log_comb_1, label='$\\lambda^k = \\lambda^{k-1} + \\tau^k Hw$, $\\lambda^0 = 0$, $\\tau^k = \\tau_02^k$')
    plt.grid()
    plt.legend()
    plt.ylabel('$\\frac{||x^{k}-x^*||}{||x^*||}$')
    plt.xlabel('Iteraciones (k)')
    plt.savefig('./images/ej4_error.png')
    
    
    iters = [len(error_log_pen_1), len(error_log_pen_2), len(error_log_mult_1), 
             len(error_log_mult_2), len(error_log_comb_1)]
    print(iters)
    plt.figure(figsize=(13,8))
    plt.bar(np.arange(len(iters)), height=iters)
    plt.ylabel('Cantidad de iteraciones (k)')
    plt.xticks(np.arange(len(iters)), labels=['Penalización cuadrática v1', 
                                              'Penalización cuadrática v2', 
                                              'Multiplicadores v1', 
                                              'Multiplicadores v2',
                                              'Combinación'])
    plt.savefig('./images/ejercicio4_iters.png')

    print()
    
    