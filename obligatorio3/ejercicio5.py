import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import itertools

def get_Pi(i, n):
    Pi = np.zeros((n, n))
    Pi[i,i] = 1
    return Pi

def get_qi(i, n):
    qi = np.zeros((n,1))
    return qi

def get_ri(i):
    return -1

def get_P_q_r(P0, q0, r0, l):
    n = len(q0)
    P = P0
    q = q0
    r = r0
    for i in range(n):
         P += l[i]*get_Pi(i,n)
         q += l[i]*get_qi(i,n)
         r += l[i]*get_ri(i)
         
    return P, q, r


def solve_PB(P0, q0, r0, verbose=True):
    n = len(q0)
    f_pb = np.inf
    for x in itertools.product([-1,1], repeat=n):
        x = np.array(x)
        cost = x.T@P0@x + 2*q0.T@x + r0
        if cost < f_pb:
            x_pb = x
            f_pb = cost 
    
    if verbose:  
        print('f_PB: ', f_pb)
        print('x_PB: ', x_pb)
    return x_pb, f_pb
            
            
def solve_DB(P0, q0, r0, verbose=True):
    pass

if __name__ == '__main__':
    # Load data   
    P0 = np.loadtxt('./obligatorio3/data/P.asc')
    q0 = np.loadtxt('./obligatorio3/data/q.asc').reshape(-1,1)
    r0 = np.loadtxt('./obligatorio3/data/r.asc').reshape(1,1)
    
        
    x_pb, f_pb = solve_PB(P0, q0, r0)
        
    
    t = cp.Variable()
    l = cp.Variable(n)
    X = cp.Variable((n+1, n+1))
    P, q, r = get_P_q_r(P0, q0, r0, l)
    
    cost = t
    constraints = [X == cp.bmat([[P, q], [q.T, r-t]]),
                   X >> 0
    ]
        
    problem = cp.Problem(cp.Maximize(cost), constraints)
    result = problem.solve(verbose=True)
    
    
    