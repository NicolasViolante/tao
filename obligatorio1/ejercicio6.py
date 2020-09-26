import numpy as np
import matplotlib.pyplot as plt

from obligatorio1.gradient_descent import ProjectedGradientDescent


class CostFunction:
    def __init__(self):
        self.input_size = 2

    def forward(self, x_in):
        x, y = x_in
        return 5*x**2 + 5*y**2 + 5*x -3*y -6*x*y + 5/4

    def backward(self, x_in):
        x, y = x_in
        grad_x = 10*x - 6*y + 5
        grad_y = 10*y - 6*x - 3
        grad = np.array([grad_x, grad_y])
        return grad

    def __call__(self, x_in):
        return self.forward(x_in)



def plot_x_log(x_log):
    x_log = np.array(x_log)
    x, y = x_log[:,0], x_log[:,1]
    plt.figure()
    plt.scatter(x,y)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    max_iter=200
    cost_function = CostFunction()
    # Parte c)
    # i) Paso decreciente
    # optimizer = ProjectedGradientDescent(cost_function, cfa_radius=0.25)
    # x = optimizer.solve(max_iter, 'line_search', verbose=True, max_step=1e-3, n_points=100)
    # plot_x_log(optimizer.x_log)

    # Parte c)
    # ii) Linea search
    optimizer = ProjectedGradientDescent(cost_function, cfa_radius=1.25)
    x = optimizer.solve(max_iter, 'decreasing', verbose=True, base=1.)
    plot_x_log(optimizer.x_log)

    print()
