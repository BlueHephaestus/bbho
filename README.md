#Description 
####(I will update this with proper LaTeX when I make a blog post, when I make a blog)

BBHO (pronounced BeeBo because names are much more fun) is a Black Box Optimization program that uses [Bayesian Optimization](https://arxiv.org/pdf/1206.2944.pdf) to optimize a given unknown (black box) function. It does this using something known as Gaussian Processes for Regression, and then what is known as an acquisition function. Here's what happens:

1. We have a function that takes n>=1 of inputs.
2. We choose two random inputs, and evaluate our black box function on these two inputs.
3. We then do the following for every point in our domain(can be whatever you choose it to be, it could for example be the ranges of two parameters on a reinforcement learning task). If n > 1 then get the cartesian product of the ranges for all hyper parameter input ranges in order to get all points on our domain.
  * a. Generate a covariance matrix K for every possible pair of points on our domain using our covariance function cov(x_i, x_j)
  * b. Using our point we are currently evaluating on the domain(the test point / x_\*), we generate a covariance vector K_* via cov(x_*, x_i) for each i in n and scalar K_** = cov(x_*, x_*)
  * c. We then get the mean u_* and variance c_* of our test point x_* with our covariance matrix K and known function outputs f: u_* = K^T_* * K^-1 * f, c_* = K_** - K^T_* * K^-1 * K_*
  * d. Store these, adding a noise parameter to our calculated variance if we have some aspect of uncertainty in our evaluations
4. Using our calculated means and variances for all inputs in the domain, compute our f_* value for each via f_* = u_* + c_* if we're maximizing or f_* = u_* - c_* if we're minimizing
5. Using our calculated means, variances, and f_* values, argmax over our acquisition function to get our next input to evaluate.
6. Evaluate our new input, and start over from step 3 until we use all our allowed evaluations.

Considering the amount of resources and/or long videos I had to watch to learn all that I now know (assuming that I didn't get something big wrong), the large amount of information needed to build from the ground-up, and the lack of a site containing all of it explained from the ground up in the way I am thinking, I plan on making a blog to contain it as well as my understanding and explanations of other concepts i've learned, and also all of my projects.

Future self will put it here: 


#Resources where I learned what I needed to go from fundamentals to this:
###Videos(These are really really helpful for ground-up explanations, and taught me most of it)

[mathematicalmonk 19.1](https://www.youtube.com/watch?v=vU6AiEYED9E)

[mathematicalmonk 19.2](https://www.youtube.com/watch?v=16oPvgOd3UI)

[mathematicalmonk 19.3](https://www.youtube.com/watch?v=clMbOOz6yR0)

[mathematicalmonk 19.4](https://www.youtube.com/watch?v=clMbOOz6yR0)

[UBC CPSC 540 Lecture #1](https://www.youtube.com/watch?v=4vGiHC35j9s)

[UBC CPSC 540 Lecture #2](https://www.youtube.com/watch?v=MfHKW5z-OOA)


###Papers and Lectures:

[General Overarching Research Paper](https://arxiv.org/pdf/1206.2944.pdf)

[Another Research Paper](https://arxiv.org/pdf/1605.07079v1.pdf)

[Harvard Slides Tutorial](https://www.iro.umontreal.ca/~bengioy/cifar/NCAP2014-summerschool/slides/Ryan_adams_140814_bayesopt_ncap.pdf)

[Good Github Repository with visualization](https://github.com/fmfn/BayesianOptimization/blob/master/examples/visualization.ipynb)

[Gaussian Process Research Paper #1](http://www.eurandom.tue.nl/events/workshops/2010/YESIV/Prog-Abstr_files/Ghahramani-lecture2.pdf)

[Gaussian Process Research Paper #2](http://courses.media.mit.edu/2010fall/mas622j/ProblemSets/slidesGP.pdf)

[Gaussian Random Vectors](http://www.rle.mit.edu/rgallager/PDFS/Gauss.pdf)

[Gaussian Process Regression](http://www.gaussianprocess.org/gpml/chapters/RW2.pdf)

[Covariance Functions #1](http://www.gaussianprocess.org/gpml/chapters/RW4.pdf)

[Covariance Functions #2](http://gpss.cc/gpip/slides/rasmussen.pdf)

[Acquisition Functions](http://www.cse.wustl.edu/~garnett/cse515t/files/lecture_notes/12.pdf)




