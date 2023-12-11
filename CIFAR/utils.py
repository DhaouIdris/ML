import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd


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



def plot_confusion_matrix(modell, valid_lod, classes):
    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in valid_lod:
        output = modell(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # Save Prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # Save Truth

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
        index=[i for i in classes],
        columns=[i for i in classes],
    )
    plt.figure(figsize=(12, 7))
    sns.heatmap(df_cm, annot=True)
    plt.savefig("confusion_matrix.png")


