import numpy
import sympy
from sympy.utilities.lambdify import lambdify

def init_calc(T, X, NU):
    
    t, x, nu = sympy.symbols('t x nu')
    
    phi = (sympy.exp(-(x - 4 * t)**2 / (4 * nu * (t + 1))) + sympy.exp(-(x - 4 * t - 2 * sympy.pi)**2 / (4 * nu * (t + 1))))
    
    phiprime = phi.diff(x)
    
    u = -2 * nu * (phiprime / phi) + 4
    
    ufunc = lambdify((t, x, nu), u, 'numpy')

    u = numpy.asarray([ufunc(T, X0, NU) for X0 in X])

    return u