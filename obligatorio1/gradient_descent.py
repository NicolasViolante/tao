import numpy as np


class GradientDescent():
    def __init__(self, objetive):
        self.objetive = objetive
        self.x_log = []
    
    def _gradient_descent_step(self, step, x, k, **kwargs):
        grad_x = self.objetive.backward(x)
        alpha = self._get_current_step(step, x, k, **kwargs)
        x = x - alpha*grad_x 
        return x

    def solve(self, max_iter, step=1e-4, verbose=False, **kwargs):
        
        x = self._initialize()
        
        for k in range(max_iter):
            x = self._gradient_descent_step(step, x, k, **kwargs)
            self.x_log.append(x)
            # if _stop_condition_met():
                # break
            
        if verbose:
            print('Optimal solution is x = {} '.format(x))

        return x
        
    def _get_current_step(self, step, x, k, **kwargs):
        if isinstance(step, float):
            return step
        elif step =='arminjo':
            return _arminjo(x, **kwargs)
        elif step == 'decreasing':
            return self._decreasing(k, **kwargs)
        elif step == 'line search':
            return _line_search(**kwargs)
        else:
            raise ValueError
            
    
    def _initialize(self):
        """
        Initializes x as a vector of shape (N,) with samples drawn from a unit normal gaussian.
        """
        return np.ones(self.objetive.x_size)
        # return np.random.randn(self.A.shape[1])

    def _stop_condition_met(self):
        return False

    def _decreasing(self, k, base=0.001):
        return base / (k+1)

        
class ProjectedGradientDescent(GradientDescent):
    def __init__(self):
        super().__init__()