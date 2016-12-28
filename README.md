#Description 

BBHO (pronounced BeeBo because names are much more fun) is a Black Box Optimization program that uses [Bayesian Optimization](https://arxiv.org/pdf/1206.2944.pdf) to optimize a given unknown (black box) function. It does this using something known as Gaussian Processes for Regression, and then what is known as an acquisition function. To get an entire overarching and fundamental understanding of this method, 

###[I've made a series on my blog detailing it.](https://dark-element.com/2016/10/10/bayesian-optimization-of-black-box-functions/)

###[Resources & Sources where I learned what I needed to go from fundamentals to this](https://dark-element.com/2016/10/14/bayesian-optimization-of-black-box-functions-appendix-and-sources-resources/#resources-sources)


#Installation 

1. Clone this repository
2. Add in your own handler for whatever you are optimizing in the `black_box_functions.py` file, and how it is called in the beginning of `bbho.py`. In there, you can see several examples of different models I have previously optimized with BBHO. Each is represented as a Python class, with methods __init__() for the initialization, and evaluate() for feeding in new inputs. You can change how it is initialized to take whatever arguments you need, however I do not recommend changing the format for the evaluate() method. 
3. BBHO will pass the index of the evaluation number `bbf_evaluation_i`, the total evaluation number `bbf_evaluation_n` and finally a list of the hyper parameters `next_input`. You can choose to do nothing with `bbf_evaluation_i` and `bbf_evaluation_n`, however they are meant to be used to print the progress of the optimizer. 
4. Use the `next_input` variable as the inputs to whatever problem you are trying to optimize. In my case, I might have a keras neural network that assigns the mini batch size and learning rate like `mini_batch_size, learning_rate = next_input`, where in this case `next_input = [some_mini_batch_size_value, some_learning_rate_value]`
5. After you have fed this input in, just make sure that your handler returns a scalar. In my configurer for lira, I get the validation accuracy and then use `np.mean(config_output)` to get one value representing how well the inputs performed. I then return it, and so my handler's evaluate method only needs 4 lines.
6. With this, you should be able to start optimizing, unfortunately there are some parameters for BBHO itself.

#Configuration

1. In my lira black box function, I initialize it with `epochs` and `run_count` variables. These represent the number of times to loop over the training set, and the number of times to run an entire training iteration, respectively. As said in the Installation instructions, you can change the initialization to respect whatever your black box function may be, but you do have to initialize what you are optimizing in the beginning of `bbho.py`.
2. After this, the remaining parameters have to do with the optimization:

`detail_n`: the number of intervals for each hyper parameter optimized, so that if you assign the `hps` variable later to be `[HyperParameter(0, 100)]`, and set `detail_n = 100`, that hyper parameter's range will be `1, 2, 3, ... 99`. If you were to set `detail_n = 50`, it would be something like `1, 3, 5, ..., 49`, and so on.

`maximizing`: a boolean for if we are maximizing or minimizing. I have not tested it extensively on minimizing, use at your own risk or do a slight transformation on the output of your handler to keep this True.

`bbf_evaluation_n`: the number of points evaluated with BBHO, not including the two random points it starts with. Changing this will change the time it takes to finish optimization.

`acquisition_function`: the acquisition function we will use later on. I have it set to the upper confidence bound acquisition function with k = confidence interval = 1.5, but feel free to change this to the options available in `acquisition_functions.py`. Further explanation of acquisition functions can be found in my blog series, [Part Three](https://dark-element.com/2016/10/13/bayesian-optimization-of-black-box-functions-part-3/) and [Appendix](https://dark-element.com/2016/10/14/bayesian-optimization-of-black-box-functions-appendix-and-sources-resources/). I have added some parameters for the exponential decay of the confidence interval, but this is for personal experiments of my own and I can not make any guarantees as to the efficacy of using a decay rate, yet.

`covariance_function`: the covariance function we will use later on. It is initially set to the matern 5/2 covariance function with lengthscale 1, but feel free to change this to the options available in `covariance_functions.py`. Further explanation of covariance functions can be found in my blog series, [Part Three](https://dark-element.com/2016/10/13/bayesian-optimization-of-black-box-functions-part-3/) and [Appendix](https://dark-element.com/2016/10/14/bayesian-optimization-of-black-box-functions-appendix-and-sources-resources/).

`hps`: mentioned earlier, this is a list of Hyper Parameter classes, found in `hyperparameter.py`. You should specify the Hyper Parameters according to the syntax `HyperParameter(min, max)`, with `min`, `max` according to the range of that hyper parameter. You should format the number of these according to the number of arguments your handler is ready for. If I was prepared for mini batch size and regularization rate in my handler, I might have `hps = [HyperParameter(0, 50), HyperParameter(0, 1)]` 


Feel free to contact me with any questions or help, and Good luck, have fun!

