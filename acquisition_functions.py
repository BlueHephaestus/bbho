import numpy as np

import bbho_base
from bbho_base import *

#Initialize the theano functions here so we don't have to initialize 
    #them every time we call the function(once every evaluation)

means = T.vector()
variances = T.vector()
values = T.vector()
cdfs = T.vector()
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

        self.f = theano.function([means, stddevs, variances, values], 
                    outputs = T.argmax(stddevs * (gaussian_distribution(values, means, variances) * cdf(values, means, stddevs) + gaussian_distribution(values, 0, 1))),
                    allow_input_downcast=True
                    )

    def evaluate(self, means, variances, values):
        return self.f(means, self.stddev(variances), variances, values)


class upper_confidence_bound(acquisition_function):

    def __init__(self, confidence_interval):
        acquisition_function.__init__(self, confidence_interval)
        self.f = theano.function([means, variances], 
                    outputs = T.argmax(means + self.confidence_interval * T.sqrt(variances)),
                    allow_input_downcast=True
                    )
   
    def evaluate(self, means, variances, values):
        return self.f(means, variances)

