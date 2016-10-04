#Description (I will update this when I learn how to LaTeX)

BBHO (pronounced BeoBo because names are much more fun) is a Black Box Optimization program that uses [Bayesian Optimization](https://arxiv.org/pdf/1206.2944.pdf) to optimize a given unknown (black box) function. It does this using something known as Gaussian Processes for Regression, and then what is known as an acquisition function. Here's what happens:

1. We have a function that takes ![](http://www.sciweavers.org/tex2img.php?eq=%5C%5Bn%20%5Cgeq%201%5C%5D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) of inputs. (I need to learn how to latex)
2. We choose two random inputs, and evaluate our black box function on these two inputs.
3. We then do the following for every point in our domain(can be whatever you choose it to be, it could for example be the ranges of two parameters on a reinforcement learning task). If ![](http://www.sciweavers.org/tex2img.php?eq=%5C%5Bn%20%3E%201%5C%5D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0), then get the cartesian product of the ranges for all hyper parameter input ranges in order to get all points on our domain.
  * a. Generate a covariance matrix ![](http://www.sciweavers.org/tex2img.php?eq=%5C%5BK%20%5C%5D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) for every possible pair of points on our domain(notating this as K)
  * b. Using our point we are currently evaluating on the domain(the test point), we generate a covariance vector ![](http://www.sciweavers.org/tex2img.php?eq=%5C%5BK_%2A%20%5C%5D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
  * c. ![](http://www.sciweavers.org/tex2img.php?eq=%5C%5Bn%20%3E%201%5C%5D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)



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



