import numpy as np


class GradientDescent:
    def __init__(self, cost_function):
        self.cost_function = cost_function
        self.x_log = []

    def solve(self, max_iter, strategy, verbose=False, **kwargs):

        x = self._initialize()
        self.x_log.append(x)
        learning_step_strategy = self._get_learning_step_strategy(strategy, **kwargs)

        for k in range(max_iter):
            x = self._gradient_descent_step(x, learning_step_strategy)

            if self._is_stop_condition_met():
                break

        if verbose:
            print(f'Optimal solution is x = {x}. {k+1} iterations')

        return x


    def _gradient_descent_step(self, x, learning_step_strategy):
        grad_x = self.cost_function.backward(x)
        direction = -grad_x
        learning_step = learning_step_strategy.get_step(f=self.cost_function,
                                                        x=x,
                                                        grad_x=grad_x,
                                                        direction=direction)

        x = self._update_position(x, learning_step, direction)

        self.x_log.append(x)
        return x

    @staticmethod
    def _update_position(x, learning_step, direction):
        return x + learning_step*direction

    @staticmethod
    def _get_learning_step_strategy(strategy, **kwargs):
        step_strategies = {'fixed': FixedStep,
                           'decreasing': DecreasingStep,
                           'armijo': ArmijoStep,
                           'line_search': LineSearchStep}
        if strategy not in step_strategies:
            raise ValueError(f'Unknown learning step strategy: {strategy}')
        return step_strategies[strategy].create(**kwargs)

    def _initialize(self):
        return np.zeros(self.cost_function.input_size)

    def _is_stop_condition_met(self, min_improvement=1e-8):
        if len(self.x_log) >= 2:
            improvement = np.linalg.norm(self.x_log[-1] - self.x_log[-2])
            return improvement <= min_improvement
        return False



class ProjectedGradientDescent(GradientDescent):
    def __init__(self, cost_function, cfa_radius):
        super().__init__(cost_function)
        self.cfa_radius = cfa_radius

    def _update_position(self, x, learning_step, direction):
        return self.project_to_cfa(x + learning_step*direction)

    def project_to_cfa(self, x):
        if np.linalg.norm(x) > self.cfa_radius:
            return (self.cfa_radius*x) / np.linalg.norm(x)
        return x


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
        f = kwargs['f']
        x = kwargs['x']
        direction = kwargs['direction']
        grad_x = kwargs['grad_x']

        step = self.max_step
        while not self._is_step_good(step, f, x, grad_x, direction):
            step *= self.beta
        return step

    def _is_step_good(self, step, f, x, grad_x, direction):
        improvement = f(x) - f(x + step*direction)
        min_improvement = -self.sigma*step*np.dot(direction, grad_x)
        return improvement >= min_improvement


class LineSearchStep(LearningStep):
    def __init__(self, max_step, n_points):
        self.steps = np.linspace(0, max_step, n_points)

    @classmethod
    def create(cls, **kwargs):
        max_step = kwargs['max_step']
        n_points = kwargs['n_points']
        return cls(max_step, n_points)

    def get_step(self, **kwargs):
        f = kwargs['f']
        x = kwargs['x']
        direction = kwargs['direction']

        steps_in_direction = [step*direction for step in self.steps]
        f_values_in_search_range = np.array([f(x+step_in_direction) for step_in_direction in steps_in_direction])
        step = self.steps[np.argmin(f_values_in_search_range)]
        return step

