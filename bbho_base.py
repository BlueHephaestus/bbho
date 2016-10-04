import numpy as np
import itertools#For Cartesian product 

def gaussian_distribution(x, mean, stddev):
    #Gets input x, mean, and variance
    #Returns vector or scalar from input
    return (np.exp(-((x-mean)**2)/(2*stddev))) / (np.sqrt(2*stddev*np.pi))

def cdf(x, mean, variance):
    #Get values to compute cdf over
    dist_values = gaussian_distribution(np.arange(x-100, x, .1), mean, variance)
    
    #Equivalent to the last element of cumulative sum
    return sum(dist_values)

def get_cov_matrix(f, cov):
    #Given a vector f, generate the covariance matrix 
    #f because known inputs
    f_n = len(f)
    f_m = np.zeros(shape=(f_n, f_n))
    for row_i, f_i in enumerate(f):
        for col_i, f_j in enumerate(f):
            f_m[row_i, col_i] = cov.evaluate(f_i, f_j)

    return f_m

def get_cov_vector(f, test_f, cov):
    #Given a vector f and scalar f* (test_f)
    #Generate a covariance vector for each value in f
    f_n = len(f)
    f_v = np.zeros(shape=(f_n, 1))
    for row_i, f_i in enumerate(f):
        f_v[row_i] = cov.evaluate(test_f, f_i)

    return f_v

def cartesian_product(vectors):
    return [i for i in itertools.product(*vectors)]
