import logging
import sys
import os
import pathlib

import yaml
import torch
import torchinfo.torchinfo as torchinfo

import models
import Optim_Loss
import dataloader
import utils




def train(config):
   
   use_cuda = torch.cuda.is_available()
   device = torch.device("cuda") if use_cuda else torch.device("cpu")

   logging.info("= Building the dataloaders")
   train_lod, valid_lod, inputsize, numclasses, classes = dataloader.get_dataloaders(config, False)
   model = config["model"]["class"]


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


       # build a moodel
   logging.info("= Building the model")
   modell= models.build_model(config, inputsize, numclasses)
   modell = modell.to(device)

    # build a optimizer, loss
   logging.info("= Building the loss function")
   loss = Optim_Loss.loss_function(config)

   logging.info("= Building the optimizers")
   optim = Optim_Loss.get_optimizer(config, modell.parameters())
   scheduler=Optim_Loss.get_scheduler(optim)
   # définition du checkpoint
   logging.info("= Building the checkpoint")

       # définition du checkpoint
   logging.info("= Building the checkpoint")

   logdir = utils.generate_unique_logpath(
        config["logging"]["logdir"], config["model"]["class"]
    )
  
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   model_checkpoint = utils.ModelCheckpoint(modell, logdir + "/best_model.pt")

   epochs = config['nepochs']
   for _ in range(epochs):
      train_loss = utils.train(model, loss, optim, device, train_lod)
      valid_loss = utils.test(model, loss, device, valid_lod)
