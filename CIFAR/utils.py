import os
import matplotlib.pyplot as plt
import numpy as np


class ModelCheckpoint:
    def __init__(self, model, filepath):
        self.min_loss = None
        self.model = model
        self.filepath = filepath

    def update(self, loss):
        if self.min_loss is None or loss < self.min_loss:
            self.min_loss = loss
            torch.save(self.model.state_dict(), self.filepath)
            print("saving a better model")
            self.min_loss = loss


def generate_unique_logpath(logdir, name):
    i = 0
    while True:
        run_name = name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1

