# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib

# External imports
import yaml
import wandb
import torch
import torch.nn as nn
import torchinfo.torchinfo as torchinfo
import numpy as np
from tqdm import tqdm


# Local imports
import data
import models
import optim
import utils

def train(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    if use_cuda:
        print("using gpu")
    else:
        print("using cpu")

    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
        wandb_log = wandb.log
        wandb_log(config)
        logging.info(f"Will be recording in wandb run name : {wandb.run.name}")
    else:
        wandb_log = None

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(
        data_config, use_cuda
    )
    
    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size[0], 1)
    model.load_state_dict(torch.load("/usr/users/sdim/sdim_15/planktonDL/model_logs/UnetPlus_5/best_model.pt")) 
    model.to(device)

    # Build the loss
    logging.info("= Loss")
    loss = optim.get_loss(config["loss"]["name"], config)

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = optim.get_optimizer(optim_config, model.parameters())

    # Build the callbacks
    logging_config = config["logging"]
    # Let us use as base logname the class name of the modek
    logname = model_config["class"]
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    # Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    # Make a summary script of the experiment
    input_size = next(iter(train_loader))[0].shape
    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log is not None else "")
        + "## Summary of the model architecture\n"
        + f"{torchinfo.summary(model, input_size=input_size)}\n\n"
        + "## Loss\n\n"
        + f"{loss}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_loader.dataset.dataset}\n"
        + f"Validation : {valid_loader.dataset.dataset}"
    )
    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)
    if wandb_log is not None:
        wandb.log({"summary": summary_text})

    # Define the early stopping callback
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=False
    )

    valid_iter = None
    for e in range(config["nepochs"]):
        # Train 1 epoch
        train_loss, train_metrics = utils.train(model, train_loader, loss, optimizer, device, config)

        # Test
        test_loss, test_metrics = utils.test(model, valid_loader, loss, device, config)

        updated = model_checkpoint.update(test_metrics["f1"])
        logging.info(
            "[%d/%d] Test F1-score : %.3f %s"
            % (
                e,
                config["nepochs"],
                test_metrics["f1"],
                "[>> BETTER <<]" if updated else "",
            )
        )

        # Update the dashboard
        metrics = {"train_CE": train_loss, "test_CE": test_loss,
                   **{"train_"+k:v for k,v in train_metrics.items()},
                   **{"test_"+k:v for k,v in test_metrics.items()}}
        if wandb_log is not None:
            logging.info("Logging on wandb")
            wandb_log(metrics)
            valid_iter = utils.visualize_predictions(model, valid_loader, device, config, valid_iter=valid_iter, n_samples=4)



def test(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    if use_cuda:
        print("using gpu")
    else:
        print("using cpu")

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    test_loader, input_size, num_classes = data.get_test_dataloaders(
        data_config, use_cuda
    )

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, 1, 1)
    model.load_state_dict(torch.load("model_logs/UnetPlus_0/best_model.pt"))
    model.to(device)

    # Inference
    logging.info("= Running inference on the test set")
    reconstructed_images = {}  # Stores final full-sized predictions
    normalization_map = {}  # Stores patch counts for averaging

    patch_size = test_loader.dataset.patch_size
    gaussian = utils.gaussian_kernel(patch_size, sigma=128.)

    with torch.no_grad():
        for batch in tqdm(test_loader):
            images, row_starts, col_starts, img_indices = batch
            images = images.to(device).to(dtype=torch.float32)

            # Forward pass (keep raw logits)
            outputs = model(images).cpu().numpy()

            for i in range(outputs.shape[0]):
                img_idx = img_indices[i].item()
                row_start = row_starts[i].item()
                col_start = col_starts[i].item()
                logit_patch = outputs[i][0]  # Extract 2D logits
                width, height = test_loader.dataset.image_sizes[img_idx]

                if img_idx not in reconstructed_images:
                    reconstructed_images[img_idx] = np.zeros((height, width), dtype=np.float32)
                    normalization_map[img_idx] = np.zeros((height, width), dtype=np.float32)

                # Define patch end coordinates
                row_end = min(row_start + patch_size, reconstructed_images[img_idx].shape[0])
                col_end = min(col_start + patch_size, reconstructed_images[img_idx].shape[1])

                valid_patch_height = row_end - row_start
                valid_patch_width = col_end - col_start

                # Accumulate logits and count overlapping contributions
                weight_patch = gaussian[:valid_patch_height, :valid_patch_width]
                reconstructed_images[img_idx][row_start:row_end, col_start:col_end] += logit_patch[:valid_patch_height, :valid_patch_width] * weight_patch
                normalization_map[img_idx][row_start:row_end, col_start:col_end] += weight_patch

                # reconstructed_images[img_idx][row_start:row_end, col_start:col_end] += logit_patch[:valid_patch_height, :valid_patch_width]
                # normalization_map[img_idx][row_start:row_end, col_start:col_end] += 1

    #logging.info("= Generating submission file")
    for img_idx in reconstructed_images:
        reconstructed_images[img_idx] /= normalization_map[img_idx]  # Average overlapping regions
        reconstructed_images[img_idx] = (torch.sigmoid(torch.tensor(reconstructed_images[img_idx])) >= model_config['threshold']).byte().numpy()



if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")
