"""
Penultimate program - making to port over to full on machine learning applications after testing here.

Still working on a name for the full one, it won't be CHO MK 2 since it now uses a conventional hyper 
  parameter tuning method, but I don't want to call it just Bayesian Optimization (BO) since that would
  omit some details, like hyper parameters tuning. I'll figure it out, but for now, this program works like this:

Given an input function,

1. Get input domains via xrange(min, max, ((max-min)/n)) where n is arbitrary level of detail, i.e. 10,000 
    and min/max are defined beforehand for each input
2. Generate two random input vectors x1, x2 where each element is min <= x <= max and on our domain
3. Evaluate x1 & x2 to get f1 & f2
4. Generate n test means(u* s) and test variances(c* s) across domain via the following for each test input x*:
    a. Generate covariance matrix with known inputs
    b. Generate covariance vector and transpose with known inputs and test input, as well as diagonal
    c. u* = (K* transpose) * (K inverse) * (known_evaluations)
    d. c* = (K* diag) - (K* transpose) * (K inverse) * (K*)
5. 
    If maximizing, set output at each test x (f(x*)) = u* + c*
    If minimizing, set f(x*) = u* - c*

6. Generate upper confidence bound acquisition function vector and get best new inputs via argmaxing it:
    x_i = domain[argmax(u* - confidence_interval * c*)

7. Go back to step 3 until i == end_iteration
8. Final inputs = domain[argmax(f*)]
"""
import sys, time

import numpy as np

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

import hyperparameter
from hyperparameter import HyperParameter

import acquisition_functions
from acquisition_functions import *

import covariance_functions 
from covariance_functions import *

import bbho_base
from bbho_base import *

#Our test black box optimization functions
"""
def bbf(x):
   #return np.sin(x**(2.5/2.0))+(x/3.6)**2
   #return (x-7)**3 -3.5*(x-7)**2 + 4
"""

def bbf(x):
    #return -(x-2)**2
    return np.exp(-(x-2)**2) + np.exp(-((x-6)**2)/10.0) + (1.0/(x**2 + 1))
    #return np.cos(x[0])*np.cos(x[1])*np.exp(-((x[0]-np.pi)**2 + (x[1]-np.pi)**2))
    #return -(x[0]-2)**2 - (x[1]-3)**2 + 4
    #return -(x[0]-2)**2 - (x[1]-3)**2 - (x[2]-1)**2+ 4

#For efficiency comparisons
start_time = time.time() 
        
#Number of evaluated input points / level of detail
detail_n = 30

#If we want the highest point or lowest point
maximizing = True

#Number of bbf evaluations allowed to perform before ending optimization
bbf_evaluation_n = 20

#Choice of acquisition function and af parameters
confidence_interval = 1.5 
acquisition_function = upper_confidence_bound(confidence_interval)

#Choice of covariance function and cf parameters
lengthscale = 1.0
v = [5/2.0]
covariance_function = squared_exponential(lengthscale, v)

#Initialize ranges for each parameter into a resulting matrix
hps = [HyperParameter(0, 10)]

#UI settings
plot_results = True
plot_3d_results = False

"""END TUNABLE PARAMETERS"""

#Initialize independent domains of each parameter
#We will do the cartesian product on the vectors contained here to get our entire sets of multidimensional inputs
independent_domains = np.array([np.arange(hp.min, hp.max, ((hp.max-hp.min)/float(detail_n))) for hp in hps])

#Get the total number of outputs as n^r
n = detail_n**len(hps)

#Get the cartesian product of all vectors contained to get entire multidimensional domain
domain = [np.array(i) for i in cartesian_product(independent_domains)]

#Get our axis vectors if plotting
if plot_results:
    domain_x = np.copy(independent_domains[0])
elif plot_3d_results:
    domain_x = np.copy(independent_domains[0])
    domain_y = np.copy(independent_domains[1])

#Now that we have full domain, we can shuffle the original to get two random input vectors
for independent_domain in independent_domains:
    np.random.shuffle(independent_domain)

#Get our different values easily by transposing
x1, x2 = independent_domains.transpose()[:2]

#Now that we have our two random input vectors, evaluate them and store them in our bbf inputs and outputs vector
#Modify the bbf function when you make this more complicated with input to a bot
#This needs to not be a np.array since we have to append
bbf_inputs = [x1, x2]

#This needs to be np array so we can do vector multiplication
bbf_evaluations = np.array([bbf(x1), bbf(x2)])

#Our main loop to go through every time we evaluate a new point, until we have exhausted our allowed 
#   black box function evaluations.
for bbf_evaluation_i in range(2, bbf_evaluation_n):
    sys.stdout.write("\rDetermining Point #%i" % (bbf_evaluation_i+1))
    sys.stdout.flush()
    #print "Determining Point #%i" % (bbf_evaluation_i+1)

    #Since we reset this every time we generate through the domain
    test_means = np.zeros(shape=(n))
    test_variances = np.zeros(shape=(n))
    test_values = np.zeros(shape=(n))

    for test_input_i, test_input in enumerate(domain):
        #Generate our covariance matrices and vectors
        training_cov_m = get_cov_matrix(bbf_inputs, covariance_function)#K
        
        #Clip a small amount so we don't have singular matrix
        training_cov_m = training_cov_m + (np.eye(training_cov_m.shape[0])*1e-7)
        training_cov_m_inv = np.linalg.inv(training_cov_m)#K^-1
        test_cov = get_cov_vector(bbf_inputs, test_input, covariance_function)#K*
        test_cov_T = test_cov.transpose()#K*T
        test_cov_diag = covariance_function.evaluate(test_input, test_input)#K**

        #Compute test mean using our Multivariate Gaussian Theorems
        #print test_cov_T.shape, training_cov_m_inv.shape, bbf_evaluations.shape
        test_mean = np.dot(np.dot(test_cov_T, training_cov_m_inv), bbf_evaluations)

        #Compute test variance using our Multivariate Gaussian Theorems
        test_variance = test_cov_diag - np.dot(np.dot(test_cov_T, training_cov_m_inv), test_cov)

        #Store them for use with our acquisition function
        test_means[test_input_i] = test_mean
        test_variances[test_input_i] = test_variance + 0.01

    #Now that we have all our means u* and variances c* for every point in the domain,
    #Move on to determining next point to evaluate using our acquisition function
    #If we want the point that will give us next greatest input, do u + c, otherwise u - c
    if maximizing:
        test_values = test_means + test_variances
    else:
        test_values = test_means - test_variances

    #Set the filename
    fname = "results/%s" % str(bbf_evaluation_i)

    #Plot our updates
    #Make these in seperate file once we have 3d working
    if plot_results:
        plt.plot(domain_x, test_means)
        plt.plot(bbf_inputs, bbf_evaluations, 'bo')
        plt.plot(domain_x, test_means+test_variances, 'r')
        plt.plot(domain_x, test_means-test_variances, 'r')
        plt.plot(domain_x, bbf(domain_x), 'y')
        plt.savefig("%s.jpg" % fname, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches='tight', pad_inches=0.1,
            frameon=None)
        plt.gcf().clear()
    elif plot_3d_results:
        if bbf_evaluation_i == bbf_evaluation_n-1:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            #X & Y have to be matrices of all vertices
            #Z has to be matrix of outputs
            #Convert our vectors to compatible matrix counterparts
            Y = np.array([[i] for i in domain_y])

            X = np.tile(domain_x, (detail_n, 1))
            Y = np.tile(Y, (1, detail_n))

            #This ones easy, just reshape
            #print X.shape, Y.shape, test_means.shape
            Z1 = test_means.reshape(detail_n, detail_n)
            Z2 = test_variances.reshape(detail_n, detail_n)
            Z3 = (test_means + test_variances).reshape(detail_n, detail_n)
            Z4 = (test_means - test_variances).reshape(detail_n, detail_n)


            ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap=cm.coolwarm)
            #ax.plot_wireframe(X, Y, Z2, rstride=1, cstride=1)
            ax.plot_wireframe(X, Y, Z3, rstride=1, cstride=1)
            ax.plot_wireframe(X, Y, Z4, rstride=1, cstride=1)
            plt.show()

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
    bbf_evaluations.append(bbf(next_input))
    bbf_evaluations = np.array(bbf_evaluations)

best_input = bbf_inputs[np.argmax(bbf_evaluations)]
print ""
print bbf_evaluations
print "Best input found after {} iterations: {}".format(bbf_evaluation_n, best_input)
print "Time to run: %f" % (time.time() - start_time)
