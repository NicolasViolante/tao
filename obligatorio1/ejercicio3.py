import numpy as np
import matplotlib.pyplot as plt


class Function:
    def __init__(self, f, domain, name):
        self.f = f
        self.domain = domain
        self.name = name

    def save_function_plot(self, title, show=False):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(self.domain, self.f(self.domain))
        ax.grid()
        ax.set_xlabel('x')
        ax.set_ylabel('function(x)')
        ax.set_title(title)
        fig.savefig('./images/ejercicio3_{}.png'.format(self.name))
        if show:
            plt.show()


if __name__ == "__main__":
    # Part a)
    #-------------------------------------------------------------------------------------------------------------------
    f1 = Function(f=lambda x: 4*x**4 - x**3 - 4*x**2 + 1,
                  domain=np.linspace(-1.0, 1.0, 500),
                  name='f1')
    f1.save_function_plot(title="$4x^4 -x^3 -4x^2 + 1$")

    # Part b)
    # -------------------------------------------------------------------------------------------------------------------
    f2 = Function(f=lambda x:x**3,
                  domain=np.linspace(-1.0, 1.0, 500),
                  name='f2')
    f2.save_function_plot(title="$x^3$")

    # Part c)
    # -------------------------------------------------------------------------------------------------------------------
    a = 0.5
    f3 = Function(f=lambda x: (x-a)**2 + 1,
                  domain=np.linspace(-1.0, 1.0, 500),
                  name='f3_a={}'.format(a))
    f3.save_function_plot(title="$(x-{})^2+1$".format(a))

    a = 4
    f3.f = lambda x: (x-a)**2 + 1
    f3.name = 'f3_a={}'.format(a)
    f3.save_function_plot(title="$(x-{})^2+1$".format(a))

    # Part d)
    # -------------------------------------------------------------------------------------------------------------------

    print()


