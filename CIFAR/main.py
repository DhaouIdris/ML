import logging
import sys
import os
import pathlib

import yaml
import torch
import torchinfo.torchinfo as torchinfo

import Optim_Loss
import dataloader
import utils



def train(config):
   
   use_cuda = torch.cuda.is_available()
   device = torch.device("cuda") if use_cuda else torch.device("cpu")

   logging.info("= Building the dataloaders")
   train_lod, valid_lod, inputsize, numclasses, classes = dataloader.get_dataloaders(config, False)
   model = config["model"]["class"]

   epochs = config['nepochs']
      for _ in range(epochs):
         train_loss = utils.train(model, loss, optim, device, train_lod)
         valid_loss = utils.test(model, loss, device, valid_lod)


       # creer un datalaoder a parti du fichier config
   logging.info("= Building the dataloaders")
   (
        train_lod,
        valid_lod,
        inputsize,
        numclasses,
        classes,
        _,
        _,
        _,
   ) = dataloader.get_dataloaders(config, False)


