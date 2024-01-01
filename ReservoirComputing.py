from rctorch import *
import torch

fp_data = data.load("forced_pendulum", train_proportion = 0.2)

force_train, force_test = fp_data["force"]
target_train, target_test = fp_data["target"]

hps = {'connectivity': 0.4,
           'spectral_radius': 1.13,
           'n_nodes': 202,
           'regularization': 1.69,
           'leaking_rate': 0.0098085,
           'bias': 0.49}

my_rc = RcNetwork(**hps, random_state = 210, feedback = True, activation_function= "relu")
my_rc.fit(y = target_train) #training
