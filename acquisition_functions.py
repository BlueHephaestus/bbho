import numpy as np

import bbho_base
from bbho_base import *

means = T.vector()
stddevs = T.vector()
variances = T.vector()
values = T.vector()
cdfs = T.vector()
dist_values = T.vector()
normal_dist_values = T.vector()
confidence_interval = T.scalar()

class acquisition_function(object):

    def __init__(self, confidence_interval):
        self.confidence_interval = confidence_interval

class probability_improvement(acquisition_function):

    def __init__(self, confidence_interval):
        acquisition_function.__init__(self, confidence_interval)
        self.f = theano.function([cdfs], 
                    outputs = T.argmax(cdfs),
                    allow_input_downcast=True
                    )

    def evaluate(self, means, variances, values):
        #We have to format it like this so that our cdf function does not get called until we have means, variances, and values
        #Unlike if we included this in the theano function, where it would be called with the initialization of the function
        cdfs = cdf(values, means, variances)
        return self.f(cdfs)

class expected_improvement(acquisition_function):

    def __init__(self, confidence_interval):
        acquisition_function.__init__(self, confidence_interval)
        
        #We assign this so we don't compute it twice in our function
        self.stddev = theano.function([variances], T.sqrt(variances), allow_input_downcast=True)

        self.f = theano.function([stddevs, dist_values, cdfs, normal_dist_values], 
                    outputs = T.argmax(stddevs * dist_values * cdfs + normal_dist_values),
                    allow_input_downcast=True
                    )

    def evaluate(self, means, variances, values):
        #We have to format it like this so that our cdf function does not get called until we have means, variances, and values
        #Unlike if we included this in the theano function, where it would be called with the initialization of the function
        dist_values = gaussian_distribution_v(values, means, variances)
        cdfs = cdf(values, means, variances)
        normal_dist_values = gaussian_distribution_v(values, np.zeros_like(values), np.ones_like(values))

        return self.f(self.stddev(variances), dist_values, cdfs, normal_dist_values)


class upper_confidence_bound(acquisition_function):

    def __init__(self, confidence_interval):
        acquisition_function.__init__(self, confidence_interval)
        self.f = theano.function([means, variances], 
                    outputs = T.argmax(means + self.confidence_interval * T.sqrt(variances)),
                    allow_input_downcast=True
                    )
   
    def evaluate(self, means, variances, values):
        return self.f(means, variances)

