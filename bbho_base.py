import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nlinalg import matrix_inverse, eig

import itertools#For Cartesian product 

#We initialize the following here so we don't have to when we call the functions that use them.

"Matrix inverse"
m = T.matrix()
invert_matrix = theano.function([m], matrix_inverse(m))

"Adds one axis if we are hoping to use the Prob Density function with vectors instead of matrices, as it's designed"
v = T.vector()
vector_to_column = theano.function([v], v.dimshuffle(0, 'x'))

#Multivariate covariance mean & variance
test_cov = T.matrix()
test_cov_T = T.matrix()
test_cov_diag = T.vector()
training_cov_m_inv = T.matrix()
bbf_evaluations = T.matrix()

#Gaussian Distribution vars
inputs_m = T.matrix()
means_m = T.matrix()
variances_m = T.matrix()

inputs_v = T.vector()
means_v = T.vector()
variances_v = T.vector()

"Multivariate covariance mean"
compute_mv_mean = theano.function([test_cov_T, training_cov_m_inv, bbf_evaluations], 
                outputs = T.dot(T.dot(test_cov_T, training_cov_m_inv), bbf_evaluations),
                allow_input_downcast=True
            )

"Multivariate covariance variance"
compute_mv_variance = theano.function([test_cov_diag, test_cov_T, training_cov_m_inv, test_cov], 
                outputs = test_cov_diag - T.diag(T.dot(T.dot(test_cov_T, training_cov_m_inv), test_cov)),
                allow_input_downcast=True
            )


"Probability Density Function / Gaussian Distribution Function (for matrix input)"
probability_density_function_m = theano.function([inputs_m, means_m, variances_m], 
                outputs = (T.exp(-(T.sqr(inputs_m-means_m))/(2*variances_m)) / (T.sqrt(2*variances_m*np.pi))),
                allow_input_downcast=True
            )

"Probability Density Function / Gaussian Distribution Function (for vector input)"
probability_density_function_v = theano.function([inputs_v, means_v, variances_v], 
                outputs = (T.exp(-(T.sqr(inputs_v-means_v))/(2*variances_v)) / (T.sqrt(2*variances_v*np.pi))),
                allow_input_downcast=True
            )

def gaussian_distribution_m(inputs, means, variances):
    #For matrices
    #Gets input x, means, and variances
    #Returns vector or scalar from input
    return probability_density_function_m(inputs, means, variances)

def gaussian_distribution_v(inputs, means, variances):
    #For vectors
    #Gets input x, means, and variances
    #Returns vector or scalar from input
    return probability_density_function_v(inputs, means, variances)

def cdf(inputs, means, variances):
    #Get values to compute cdf over
    #print inputs.shape, means.shape, variances.shape
    #print np.array([np.arange(input-100, input, .1) for input in inputs]).shape

    #Convert to matrix counterparts so we can correctly get an estimate of the cdf
    inputs = np.array([np.arange(input-100, input, .1) for input in inputs])

    #We just repeat these over the axis we are getting different inputs on, which 
        #is why they are repeated according to inputs.shape[1]
    means = np.array([np.repeat(mean, inputs.shape[1]) for mean in means])
    variances = np.array([np.repeat(mean, inputs.shape[1]) for variance in variances])

    dist_values = gaussian_distribution_m(inputs, means, variances)
    #dist_values = gaussian_distribution(inputs, means, variances)
    
    #Equivalent to the last element of cumulative sum
    #numpy sum is faster 
    return np.sum(dist_values, axis=1)

def get_cov_matrix(f, cov):
    #Given a vector f, generate the covariance matrix 
    #f because known inputs
    #Numpy loops faster than theano
    f_n = len(f)
    f_m = np.zeros(shape=(f_n, f_n))
    for row_i, f_i in enumerate(f):
        for col_i, f_j in enumerate(f):
            f_m[row_i, col_i] = cov.evaluate(f_i, f_j)

    return f_m

def get_cov_vector(f, test_f, cov):
    #Given a vector f and scalar f* (test_f)
    #Generate a covariance vector for each value in f
    #Numpy loops faster than theano
    f_n = len(f)
    f_v = np.zeros(shape=(f_n))
    for row_i, f_i in enumerate(f):
        f_v[row_i] = cov.evaluate(test_f, f_i)

    return f_v

def cartesian_product(vectors):
    return [np.array(i) for i in itertools.product(*vectors)]

def convert_vector_to_column(v):
    return vector_to_column(v)

def theano_matrix_inv(m):
    return invert_matrix(m)

def get_test_means(test_cov_T, training_cov_m_inv, bbf_evaluations):
    return compute_mv_mean(test_cov_T, training_cov_m_inv, bbf_evaluations)

def get_test_variances(test_cov_diag, test_cov_T, training_cov_m_inv, test_cov):
    return compute_mv_variance(test_cov_diag, test_cov_T, training_cov_m_inv, test_cov)
