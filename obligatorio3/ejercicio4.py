import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

def solve_linear_problem(R, d2, d3, verbose=True):
    # Solve problem
    p = cp.Variable(3)
    g = cp.Variable(2)
    t = cp.Variable()

    cost = g[0] + t
    constraints = [[1,0,1]@p + [-1, 0]@g == 0,
                   [0,1,1]@p + [0,1]@g == d2,
                   [1,-1,0]@p == d3,
                   [1,1,-1]@p == 0,
                   p[1] <= R,
                   -p[1] <= R,
                   g[0] >= 0,
                   g[1] >= 0,
                   t >= 0,
                   t >= 4*(g[1] - 40)
    ]
    prob = cp.Problem(cp.Minimize(cost),constraints)
    prob.solve()
    
    # Get results
    results = {}
    results["cost"] = prob.value
    
    results["g1"] = g[0].value
    results["g2"] = g[1].value
    
    results["p1"] = p[0].value
    results["p2"] = p[1].value
    results["p3"] = p[2].value
    
    results["t"] = t.value
    
    results["l1"] = constraints[0].dual_value
    results["l2"] = constraints[1].dual_value
    results["l3"] = constraints[2].dual_value
    results["l4"] = constraints[3].dual_value
    
    results["u1"] = constraints[4].dual_value
    results["u2"] = constraints[5].dual_value
    results["u3"] = constraints[6].dual_value
    results["u4"] = constraints[7].dual_value
    results["u5"] = constraints[8].dual_value
    results["u6"] = constraints[9].dual_value
    
    return results

if __name__ == '__main__':
    R = 30 # MW
    d3 = 10 # MW
    d2_range = np.arange(1,201) # MW
    
    results = []
    for d2 in d2_range:
        results.append(solve_linear_problem(R, d2, d3))
        
    
    g1 = np.array([r["g1"] for r in results])
    g2 = np.array([r["g2"] for r in results])
    p2 = np.array([r["p2"] for r in results])
    l3 = np.array([r["l3"] for r in results])
    u1 = np.array([r["u1"] for r in results])
    u2 = np.array([r["u2"] for r in results])
    cost = np.array([r["cost"] for r in results])

    # g1 g2
    plt.figure()
    plt.plot(g1, label='$g_1$')
    plt.plot(g2, label='$g_2$')
    plt.xlabel('$d_2$ (MW)')
    plt.ylabel('MW')
    plt.legend()
    plt.grid()
    plt.savefig('./obligatorio3/ej4_g1_g2.png')
    
    # p2
    plt.figure()
    plt.plot(p2, label='$p_2$')
    plt.xlabel('$d_2$ (MW)')
    plt.legend()
    plt.grid()
    plt.savefig('./obligatorio3/ej4_p2.png')
    
    #u1
    plt.figure()
    plt.plot(u1, label='$\\mu_1$')
    plt.xlabel('$d_2$ (MW)')
    plt.legend()
    plt.grid()
    plt.savefig('./obligatorio3/ej4_u1.png')
    
    #u2
    plt.figure()
    plt.plot(u2, label='$\\mu_2$')
    plt.xlabel('$d_2$ (MW)')
    plt.legend()
    plt.grid()
    plt.savefig('./obligatorio3/ej4_u2.png')
    
    #cost
    plt.figure()
    plt.plot(cost, label='costo total')
    plt.xlabel('$d_2$ (MW)')
    plt.legend()
    plt.grid()
    plt.savefig('./obligatorio3/ej4_costo.png')
    
    # l3
    plt.figure()
    plt.plot(l3, label='$\\lambda_3$')
    plt.xlabel('$d_2$ (MW)')
    plt.legend()
    plt.grid()
    plt.savefig('./obligatorio3/ej4_l3.png')
    
    
    
    print()
