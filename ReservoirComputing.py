from rctorch import *
import torch

fp_data = data.load("forced_pendulum", train_proportion = 0.2)

force_train, force_test = fp_data["force"]
target_train, target_test = fp_data["target"]
