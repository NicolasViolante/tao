import numpy as np
import matplotlib.pyplot as plt

from obligatorio1.gradient_descent import ProjectedGradientDescent

font = {'size': 16}
plt.rc('font', **font)


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


class CircleProjection:
    def __init__(self, radius):
        self.radius = radius

    def _project(self, x):
        if np.linalg.norm(x) > self.radius:
            return (self.radius*x) / np.linalg.norm(x)
        return x

    def __call__(self, x):
        return self._project(x)


class OutCircleProjection:
    def __init__(self, radius):
        self.radius = radius

    def _project(self, x):
        if np.linalg.norm(x) < self.radius:
            return (self.radius*x) / np.linalg.norm(x)
        return x

    def __call__(self, x):
        return self._project(x)


def save_trajectories(x_log, filepath, show=False):
    x_log = np.array(x_log)
    x, y = x_log[:,0], x_log[:,1]
    plt.figure(figsize=(10,8))
    plt.scatter(x,y)
    plt.plot(x,y)
    plt.grid()
    plt.title('Trayectoria de $(x^t, y^t)$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(filepath)
    if show:
        plt.show()


def save_point_distance_evolution(x_log, filepath, show=False):
    distances = np.linalg.norm(np.diff(x_log, axis=0), axis=1)

    plt.figure(figsize=(8,8))
    plt.plot(distances)
    plt.title('Evolución de la distancia entre puntos de $(x^t, y^t)$')
    plt.ylabel('$|| (x^t, y^t) - (x^{t-1}, y^{t-1}) ||$')
    plt.xlabel('t')
    plt.grid()
    plt.savefig(filepath)
    if show:
        plt.show()


def save_function_evolution(cost_function_log, filepath, show=False):
    plt.figure(figsize=(8,8))
    plt.plot(cost_function_log)
    plt.title('Evolución del costo $f$')
    plt.ylabel('$f(x^t,x^t)$')
    plt.xlabel('t')
    plt.grid()
    plt.savefig(filepath)
    if show:
        plt.show()


if __name__ == '__main__':
    max_iter = 1000
    cost_function = CostFunction()
    # Part c)
    # i) Line search
    optimizer = ProjectedGradientDescent(cost_function, CircleProjection(radius=0.25))
    x = optimizer.solve(max_iter, 'line_search', verbose=True, max_step=1., n_points=1000)
    cost_function_log = [cost_function(x) for x in optimizer.x_log]
    save_trajectories(optimizer.x_log, './images/ejercicio6_line_search_traj.png')
    save_point_distance_evolution(optimizer.x_log, './images/ejercicio6_line_search_diff.png')
    save_function_evolution(cost_function_log, './images/ejercicio6_line_search_f_log.png')

    # Part c)
    # ii) Decreasing step
    optimizer = ProjectedGradientDescent(cost_function, CircleProjection(radius=0.25))
    x = optimizer.solve(max_iter, 'decreasing', verbose=True, base=1.)
    cost_function_log = [cost_function(x) for x in optimizer.x_log]
    save_trajectories(optimizer.x_log, './images/ejercicio6_decreasing_traj.png')
    save_point_distance_evolution(optimizer.x_log, './images/ejercicio6_decreasing_diff.png')
    save_function_evolution(cost_function_log, './images/ejercicio6_decreasing_f_log.png')

    # Part d)
    optimizer = ProjectedGradientDescent(cost_function, OutCircleProjection(radius=1.))
    x = optimizer.solve(max_iter, 'line_search', verbose=True, max_step=1., n_points=1000)
    cost_function_log = [cost_function(x) for x in optimizer.x_log]
    save_trajectories(optimizer.x_log, './images/ejercicio6d_zeros_line_search_traj.png')
    save_point_distance_evolution(optimizer.x_log, './images/ejercicio6d_zeros_line_search_diff.png')
    save_function_evolution(cost_function_log, './images/ejercicio6d_zeros_line_search_f_log.png')


