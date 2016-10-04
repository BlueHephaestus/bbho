import numpy as np
import scipy.special as ss
import theano
import theano.tensor as T

x1 = T.fvector()
x2 = T.fvector()
l = T.fscalar()
v = T.scalar()

class covariance_function(object):
    #Superclass

    def __init__(self, lengthscale, v):
        self.lengthscale = lengthscale
        self.v = v

class dot_product(covariance_function):

    def __init__(self, lengthscale, v):
        covariance_function.__init__(self, lengthscale, v)
        self.f = theano.function([x1, x2], T.dot(x1.T, x2), allow_input_downcast=True)

    def evaluate(self, x_i, x_j):
        return 1 * self.f(x_i, x_j)

class brownian_motion(covariance_function):

    def __init__(self, lengthscale, v):
        covariance_function.__init__(self, lengthscale, v)
        self.f = theano.function([x1, x2], T.minimum(x1, x2), allow_input_downcast=True)

    def evaluate(self, x_i, x_j):
        return 1 * self.f(x_i, x_j)

class squared_exponential(covariance_function):

    def __init__(self, lengthscale, v):
        covariance_function.__init__(self, lengthscale, v)
        self.f = theano.function([x1, x2, l], 
                    T.exp(
                        T.dot(
                            (-1.0/T.dot(2.0, l)), 
                            T.sum(T.sqr(x1 - x2))
                        )
                    ) 
                , allow_input_downcast=True)

    def evaluate(self, x_i, x_j):
        return self.f(x_i, x_j, self.lengthscale)

class ornstein_uhlenbeck(covariance_function):

    def __init__(self, lengthscale, v):
        covariance_function.__init__(self, lengthscale, v)
        self.f = theano.function([x1, x2], 
                    T.exp(-1.0 * T.sqrt(T.dot((x1-x2).T, (x1-x2)))),
                allow_input_downcast=True)

    def evaluate(self, x_i, x_j):
        return self.f(x_i, x_j)

class periodic1(covariance_function):

    def __init__(self, lengthscale, v):
        covariance_function.__init__(self, lengthscale, v)
        self.f = theano.function([x1, x2], 
                    T.exp(-1.0 * T.sin(5.0 * np.pi * T.sum(T.sqr(x1-x2)))),
                allow_input_downcast=True)

    
    def evaluate(self, x_i, x_j):
        return self.f(x_i, x_j)
        
class matern(covariance_function):
    """
    NOT UPGRADING THIS YET SINCE THEANO DOESN'T HAVE THE BESSEL FUNCTION
    AND ALSO BECAUSE I DON'T KNOW WHY MY IMPLEMENTATION IS HORRIBLY BROKEN
        FOR THIS ONE, AS WELL
    """

    def __init__(self, lengthscale, v):
        covariance_function.__init__(self, lengthscale, v)

    def evaluate(self, x_i, x_j):
        dist = np.linalg.norm(x_i-x_j)
        return np.nan_to_num(((2**(1-self.v))/(ss.gamma(self.v))) * ((np.sqrt(2*self.v) * (dist/self.lengthscale))**self.v) * ss.kv(self.v, (np.sqrt(2*self.v) * (dist/self.lengthscale))))
