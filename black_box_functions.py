
"""
Used to setup our other programs for use with BBHO.

__init__ can take whatever argumenets you choose,
    as you can merely change how it is called in bbho.py,
evaluate() however must take the same.

-Blake Edwards / Dark Element
"""

import sys

import numpy as np

#Currently only on cartpole, will push to more general directory when finished
sys.path.append("../openai/classic_control/cartpole/src")

import policy_gradient_configurer

#For LIRA/DENNIS
sys.path.append("../dennis/dennis5/src")
import dennis_configurer

sys.path.append("../tuberculosis_project/lira/lira1/src")
import lira_configurer

#How we obtain a scalar output given our inputs, for varying functions
class policy_gradient(object):
    #We get our policy gradient result by averaging over all our average timestep results
    def __init__(self, epochs, timestep_n, run_count):
        self.run_count = run_count
        self.configurer = policy_gradient_configurer.Configurer(epochs, timestep_n)

    def evaluate(self, bbf_evaluation_i, bbf_evaluation_n, next_input):
        config_output = self.configurer.run_config(bbf_evaluation_i, bbf_evaluation_n, self.run_count, next_input[0], 0.0, next_input[1], 0.0, next_input[2], next_input[3])
        config_avg_output = np.mean(config_output)
        return [config_avg_output]

class dennis(object):
    def __init__(self, epochs, run_count):
        self.configurer = dennis_configurer.Configurer(epochs, run_count)

    def evaluate(self, bbf_evaluation_i, bbf_evaluation_n, next_input):
        config_output = self.configurer.run_config(next_input[0], next_input[1], next_input[2], next_input[3])
        config_avg_output = np.mean(config_output)
        return [config_avg_output]
        
class lira(object):
    def __init__(self, epochs, run_count):
        self.configurer = lira_configurer.Configurer(epochs, run_count)

    def evaluate(self, bbf_evaluation_i, bbf_evaluation_n, next_input):
        config_output = self.configurer.run_config(next_input[0], next_input[1], next_input[2])
        config_avg_output = np.mean(config_output)
        return [config_avg_output]


