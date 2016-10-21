"""
Read all the details on the github README (soon to be blog) here:
    https://github.com/DarkElement75/bbho

Made by Blake Edwards / Dark Element
"""

import sys, time

import numpy as np

import theano
import theano.tensor as T

import hyperparameter
from hyperparameter import HyperParameter

import output_grapher

import acquisition_functions
from acquisition_functions import *

import covariance_functions 
from covariance_functions import *

import bbho_base
from bbho_base import *

import black_box_functions

"""START TUNABLE PARAMETERS"""
#We use our policy gradient black box function
#Configure specifics in the black_box_functions file and so on
epochs = 400
timestep_n = 200
run_count = 3
bbf = black_box_functions.policy_gradient(epochs, timestep_n, run_count)

#For efficiency comparisons
start_time = time.time() 
        
#Number of evaluated input points / level of detail
detail_n = 10

#If we want the highest point or lowest point
maximizing = True

#Number of bbf evaluations allowed to perform before ending optimization
bbf_evaluation_n = 20

#Choice of acquisition function and af parameters
initial_confidence_interval = 1.5 
confidence_interval_decay_rate = -4.0/bbf_evaluation_n
acquisition_function = upper_confidence_bound()

#Choice of covariance function and cf parameters
lengthscale = 1.0
v = [5/2.0]#For matern1(not currently functional)
covariance_function = matern2(lengthscale, v)

#Initialize ranges for each parameter into a resulting matrix
#Mini batch size, learning rate, learning rate decay rate, discount factor
#Our level of detail / detail_n determines our step size for each
hps = [HyperParameter(0, 100), HyperParameter(1, 10), HyperParameter(0, 10), HyperParameter(0, 1)]
#hps = [HyperParameter(0, 100), HyperParameter(0, 10)]

#UI/graph settings
plot_2d_results = False
plot_3d_results = False

"""END TUNABLE PARAMETERS"""

#Initialize independent domains of each parameter
#We will do the cartesian product on the vectors contained here to get our entire sets of multidimensional inputs
independent_domains = np.array([np.arange(hp.min, hp.max, ((hp.max-hp.min)/float(detail_n))) for hp in hps])

#Get the total number of outputs as n^r
n = detail_n**len(hps)

#Get the cartesian product of all vectors contained to get entire multidimensional domain
domain = cartesian_product(independent_domains)

#Get our axis vectors if plotting
if plot_2d_results:
    domain_x = np.copy(independent_domains[0])
    domain_y = []

elif plot_3d_results:
    domain_x = np.copy(independent_domains[0])
    domain_y = np.copy(independent_domains[1])

#Now that we have full domain, we can shuffle the original to get two random input vectors
for independent_domain in independent_domains:
    np.random.shuffle(independent_domain)

#Get our different values easily by transposing
x1, x2 = independent_domains.transpose()[:2]

#Known inputs
training_inputs = T.vector()

#Known evaluations
training_outputs = T.vector()

#Cartesian product of the ranges of each of our hyper parameter ranges
test_domain = T.matrix()

#Now that we have our two random input vectors, evaluate them and store them in our bbf inputs and outputs vector
#Modify the bbf function when you make this more complicated with input to a bot
#This needs to not be a np.array since we have to append
bbf_inputs = [x1, x2]

#This needs to be np array so we can do vector multiplication
bbf_evaluations = np.array([bbf.evaluate(0, bbf_evaluation_n, x1), bbf.evaluate(1, bbf_evaluation_n, x2)])

#Our main loop to go through every time we evaluate a new point, until we have exhausted our allowed 
#   black box function evaluations.
for bbf_evaluation_i in range(2, bbf_evaluation_n):
    sys.stdout.write("\rDetermining Point #%i" % (bbf_evaluation_i+1))
    sys.stdout.flush()
    #print "Determining Point #%i" % (bbf_evaluation_i+1)

    #Decay our confidence interval by decay rate, 
    #   and adjust our evaluation index back accordingly to account for our first two random inputs
    confidence_interval = exp_decay(initial_confidence_interval, confidence_interval_decay_rate, bbf_evaluation_i-2)

    #Since we reset this every time we generate through the domain
    test_means = np.zeros(shape=(n))
    test_variances = np.zeros(shape=(n))
    test_values = np.zeros(shape=(n))

    #Generate our covariance matrices and vectors with theano backend
    training_cov_m = get_cov_matrix(bbf_inputs, covariance_function)#K
    
    #Clip a small amount so we don't have singular matrix
    training_cov_m = training_cov_m + (np.eye(training_cov_m.shape[0])*1e-7)

    #Invert
    training_cov_m_inv = theano_matrix_inv(training_cov_m)#K^-1

    #Get matrix by getting our vectors for each test point and combining
    test_cov_T = np.array([get_cov_vector(bbf_inputs, test_input, covariance_function) for test_input in domain])#K*
    test_cov = test_cov_T.transpose()#K*T
    
    #Get each diag for each test input
    test_cov_diag = np.array([covariance_function.evaluate(test_input, test_input) for test_input in domain])#K**

    #Compute test mean using our Multivariate Gaussian Theorems
    #We flatten so we don't have shape (100, 1), but shape (100,)
    #test_mean = np.dot(np.dot(test_cov_T, training_cov_m_inv), bbf_evaluations)
    test_means = get_test_means(test_cov_T, training_cov_m_inv, bbf_evaluations).flatten()
    
    #Compute test variance using our Multivariate Gaussian Theorems
    #test_variance = test_cov_diag - np.dot(np.dot(test_cov_T, training_cov_m_inv), test_cov)
    test_variances = get_test_variances(test_cov_diag, test_cov_T, training_cov_m_inv, test_cov)

    #Now that we have all our means u* and variances c* for every point in the domain,
    #Move on to determining next point to evaluate using our acquisition function
    #If we want the point that will give us next greatest input, do u + c, otherwise u - c
    #Numpy adds faster
    if maximizing:
        test_values = test_means + test_variances
    else:
        test_values = test_means - test_variances

    if plot_2d_results or plot_3d_results:
        output_grapher.graph_output(plot_2d_results, plot_3d_results, bbf_evaluation_i, bbf_evaluation_n, domain_x, domain_y, detail_n, test_means, bbf_inputs, bbf_evaluations, test_means+test_variances, test_means-test_variances)

    #Get the index of the next input to evaluate in our black box function
    #Since acquisition functions return argmax values
    next_input_i = acquisition_function.evaluate(test_means, test_variances, test_values)

    #Add our new input
    next_input = domain[next_input_i]
    #print "\tNew point: {}".format(next_input)
    bbf_inputs.append(np.array(next_input))

    #Evaluate new input
    #We need this as nparray for vector multiplication
    #But we need to append as well, so we have to do this.
    #Luckily, it's our smallest np array
    bbf_evaluations = list(bbf_evaluations)

    #Evaluate using our specified black box function
    bbf_evaluations.append(bbf.evaluate(bbf_evaluation_i, bbf_evaluation_n, next_input))

    bbf_evaluations = np.array(bbf_evaluations)

best_input = bbf_inputs[np.argmax(bbf_evaluations)]
print ""
print bbf_evaluations
print "Best input found after {} iterations: {}".format(bbf_evaluation_n, best_input)
print "Time to run: %f" % (time.time() - start_time)
