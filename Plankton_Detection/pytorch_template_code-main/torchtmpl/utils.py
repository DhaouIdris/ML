# coding: utf-8

# Standard imports
import os

# External imports
import torch
import torch.nn
import tqdm

import metrics

def generate_unique_logpath(logdir, raw_run_name):
    """
    Generate a unique directory name
    Argument:
        logdir: the prefix directory
        raw_run_name(str): the base name
    Returns:
        log_path: a non-existent path like logdir/raw_run_name_xxxx
                  where xxxx is an int
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


class ModelCheckpoint(object):
    """
    Early stopping callback
    """

    def __init__(
        self,
        model: torch.nn.Module,
        savepath,
        min_is_best: bool = True,
    ) -> None:
        self.model = model
        self.savepath = savepath
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score):
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score):
        return self.best_score is None or score > self.best_score

    def update(self, score):
        if self.is_better(score):
            torch.save(self.model.state_dict(), self.savepath)
            self.best_score = score
            return True
        return False


def train(model, loader, f_loss, optimizer, device, config, dynamic_display=True):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    optimizer -- A torch.optim.Optimzer object
    device    -- A torch.device
    Returns :
    The averaged train metrics computed over a sliding window
    """

    # We enter train mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.train()
    scaler = torch.GradScaler()
    accumulation_steps = 2
    total_loss = 0
    total_metrics = {
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "iou": 0,
    }
    num_samples = 0
    for i, (inputs, targets) in (pbar := tqdm.tqdm(enumerate(loader))):

        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.unsqueeze(1)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # Compute the forward propagation
            outputs = model(inputs)
            loss = f_loss(outputs, targets) / accumulation_steps

        # Backward and optimize
        scaler.scale(loss).backward()
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
        # Update the metrics
        # We here consider the loss is batch normalized
        train_metrics = metrics.compute_metrics(y_true=targets, y_pred=(torch.sigmoid(outputs) > config['model']['threshold']).int())
        for k in total_metrics:
            total_metrics[k] += inputs.shape[0] * train_metrics[k]
        total_loss += inputs.shape[0] * loss.item()
        num_samples += inputs.shape[0]
        pbar.set_description(f"Train loss : {total_loss/num_samples:.2f}, F1-score : {total_metrics['f1']/num_samples:.5f}")
        
    return total_loss / num_samples, {k: v / num_samples for k, v in total_metrics.items()}



def test(model, loader, f_loss, device, config):
    """
    Test a model over the loader
    using the f_loss as metrics
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    device    -- A torch.device
    Returns :
    """

    # We enter eval mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.eval()

    total_loss = 0
    num_samples = 0
    total_metrics = {
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "iou": 0,
    }
    for (inputs, targets) in loader:

        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.unsqueeze(1)
        # Compute the forward propagation
        outputs = model(inputs)
        loss = f_loss(outputs, targets)

        # Update the metrics
        # We here consider the loss is batch normalized
        total_loss += inputs.shape[0] * loss.item()
        test_metrics = metrics.compute_metrics(y_true=targets, y_pred=(torch.sigmoid(outputs) > config['model']['threshold']).int())
        for k in total_metrics:
            total_metrics[k] += inputs.shape[0] * test_metrics[k]
        num_samples += inputs.shape[0]
    total_metrics = {k: v / num_samples for k, v in total_metrics.items()}
    return total_loss / num_samples, total_metrics

def get_logdir(logdir):
    i = 0
    while True:
        log_path = logdir + "_" + str(i)
        # log_path = logdir + "-" + str(i)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


def visualize_predictions(model, valid_loader, device, config, valid_iter=None, n_samples=4):
    model.eval()  # Switch to evaluation mode
    
    if valid_iter is None:
        valid_iter = iter(valid_loader)
    
    try:
        images, targets = next(valid_iter)  # Get next batch
    except StopIteration:  
        valid_iter = iter(valid_loader)  # Reset if exhausted
        images, targets = next(valid_iter) 
        
    images, targets = images.to(device), targets.to(device)

    # Get model predictions
    with torch.no_grad():
        predictions = model(images)
        predictions = torch.sigmoid(predictions) >= config["model"]["threshold"]
    
    # Select a few samples to visualize
    samples = min(n_samples, images.size(0))
    fig, axes = plt.subplots(samples, 3, figsize=(12, 4*samples))
    
    if samples == 1:  # Handle batch_size=1 case
        axes = np.expand_dims(axes, axis=0)

    for i in range(samples):
        # Original image
        img = images[i]
        if images.shape[0] == 1:  # Grayscale image
            img = img.squeeze(0)  # Remove channel dimension
            axes[i, 0].imshow(img.cpu().numpy(), cmap='gray')
        else:
            axes[i, 0].imshow(img.cpu().permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis('off')
        
        # Ground truth (if segmentation or similar task)
        axes[i, 1].imshow(targets[i].cpu().numpy().squeeze(), cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
        
        # Prediction (same format as ground truth)
        axes[i, 2].imshow(predictions[i].cpu().numpy().squeeze(), cmap='gray')
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')

    # Log to wandb
    wandb.log({
        "predictions": [wandb.Image(fig)]
    })
    plt.close(fig)
    
    return valid_iter
