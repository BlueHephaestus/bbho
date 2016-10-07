import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nlinalg import matrix_inverse, eig

import itertools#For Cartesian product 

m = T.matrix()
invert_matrix = theano.function([m], matrix_inverse(m))

test_cov = T.matrix()
test_cov_T = T.matrix()
test_cov_diag = T.vector()
training_cov_m_inv = T.matrix()
bbf_evaluations = T.matrix()

#Multivariate covariance mean
compute_mv_mean = theano.function([test_cov_T, training_cov_m_inv, bbf_evaluations], 
            outputs = T.dot(T.dot(test_cov_T, training_cov_m_inv), bbf_evaluations),
            allow_input_downcast=True
            )

#Multivariate covariance variance 
compute_mv_variance = theano.function([test_cov_diag, test_cov_T, training_cov_m_inv, test_cov], 
                outputs = test_cov_diag - eig(T.dot(T.dot(test_cov_T, training_cov_m_inv), test_cov))[0],
                allow_input_downcast=True, on_unused_input='ignore'
            )

test_variance = test_cov_diag - np.dot(np.dot(test_cov_T, training_cov_m_inv), test_cov)

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
    f_v = np.zeros(shape=(f_n))
    for row_i, f_i in enumerate(f):
        f_v[row_i] = cov.evaluate(test_f, f_i)

    return f_v

def cartesian_product(vectors):
    return [np.array(i) for i in itertools.product(*vectors)]

def theano_matrix_inv(m):
    return invert_matrix(m)

def get_test_means(test_cov_T, training_cov_m_inv, bbf_evaluations):
    return compute_mv_mean(test_cov_T, training_cov_m_inv, bbf_evaluations)

def get_test_variances(test_cov_diag, test_cov_T, training_cov_m_inv, test_cov):
    return compute_mv_variance(test_cov_diag, test_cov_T, training_cov_m_inv, test_cov)
