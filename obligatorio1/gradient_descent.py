import numpy as np


class GradientDescent:
    def __init__(self, cost_function):
        self.cost_function = cost_function
        self.x_log = []

    def solve(self, max_iter, strategy, verbose=False, **kwargs):

        x = self._initialize()
        learning_step_strategy = self._get_learning_step_strategy(strategy, **kwargs)

        for k in range(max_iter):
            x = self._gradient_descent_step(x, learning_step_strategy)

        if verbose:
            print('Optimal solution is x = {} '.format(x))

        return x


    def _initialize(self):
        return np.ones(self.cost_function.input_size)


    def _gradient_descent_step(self, x, learning_step_strategy):
        grad_x = self.cost_function.backward(x)
        learning_step = learning_step_strategy.get_step(f=self.cost_function.forward,
                                                        x=x,
                                                        grad_x=grad_x)
        x = x - learning_step*grad_x

        self.x_log.append(x)
        return x

    @staticmethod
    def _get_learning_step_strategy(strategy, **kwargs):
        step_strategies = {'fixed': FixedStep,
                           'decreasing': DecreasingStep,
                           'armijo': ArmijoStep,
                           'line_search': LineSearchStep}
        if strategy not in step_strategies:
            raise ValueError('Unknown learning step strategy: {}'.format(strategy))
        return step_strategies[strategy].create(**kwargs)



class ProjectedGradientDescent(GradientDescent):
    def __init__(self):
        super().__init__()


#-----------------------------------------------------------------------------------------------------------------------

class LearningStep:
    def get_step(self, **kwargs):
        raise NotImplementedError

    @classmethod
    def create(cls, **kwargs):
        raise NotImplementedError


class FixedStep(LearningStep):
    def __init__(self, fixed_step):
        self.fixed_step = fixed_step

    def get_step(self, **kwargs):
        return self.fixed_step

    @classmethod
    def create(cls, **kwargs):
        fixed_step = kwargs['fixed_step']
        return cls(fixed_step)


class DecreasingStep(LearningStep):
    def __init__(self, base):
        self.base = base
        self.current_iteration = 1

    def get_step(self, **kwargs):
        step = self.base/(self.current_iteration+1)
        self.current_iteration += 1
        return step

    @classmethod
    def create(cls, **kwargs):
        base = kwargs['base']
        return cls(base)


class ArmijoStep(LearningStep):
    def __init__(self, sigma, beta, max_step):
        self.sigma = sigma
        self.beta = beta
        self.max_step = max_step

    @classmethod
    def create(cls, **kwargs):
        sigma = kwargs['sigma']
        beta = kwargs['beta']
        max_step = kwargs['max_step']
        return cls(sigma, beta, max_step)

    def get_step(self, **kwargs):
        x = kwargs['x']
        grad_x = kwargs['grad_x']
        f = kwargs['f']

        step = self.max_step
        while not self._is_step_good(step, f, x, grad_x):
            step *= self.beta
        return step

    def _is_step_good(self, step, f, x, grad_x):
        return f(x) - f(x-step*grad_x) >= -self.sigma*step*np.linalg.norm(grad_x)


class LineSearchStep(LearningStep):
    def __init__(self, max_step, n_points):
        self.steps = np.linspace(0, max_step, n_points)

    @classmethod
    def create(cls, **kwargs):
        max_step = kwargs['max_step']
        n_points = kwargs['n_points']
        return cls(max_step, n_points)

    def get_step(self, **kwargs):
        x = kwargs['x']
        grad_x = kwargs['grad_x']
        f = kwargs['f']

        x_search_range = [-step*grad_x for step in self.steps]
        f_values_in_search_range = np.array([f(x+step_x) for step_x in x_search_range])
        step = self.steps[np.argmin(f_values_in_search_range)]
        return step

