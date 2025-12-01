import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_angle(otf3d: np.ndarray):
    phase = np.angle(np.fft.fftshift(otf3d))
    fig, axes = plt.subplots(1,3, figsize=(19.2,10.8))
    axes = axes.ravel()
    for i in range(3):
        axes[i].imshow(phase[:, :, i], cmap="gray")
    plt.show()

def get_next_dir(name):
    i = 0
    while True:
        name_path = name + "-" + str(i)
        if not os.path.isdir(name_path):
            return name_path
        i = i + 1

def get_logdir(logdir):
    # TODO: add time
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    return logdir+"--"+timestamp


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_summary(trainer):
    # TODO: use torchinfo fro model summary
    summary = (f"## Logdir: {trainer.logdir}\n\n" + "## Dataset:\n" +
        str(getattr(trainer, "data", None)) + "\n\n" + "## Model:\n" + str(getattr(trainer, "model", None)) +
        "## Loss:\n" + str(getattr(trainer, "loss", None)) + "\n\n" +
        "## Optimizer:\n" + str(getattr(trainer, "optimizer", None)) +
        "## Learning rate scheduler:\n" + str(getattr(trainer, "lr_scheduler", None)) +
        "## Metrics:\n" + str(getattr(trainer, "metric_funcs", None)) +
        f"\n\n## Seed: {trainer.config['seed']}" if "seed" in trainer.config else "" +
        f"\n\n## Epochs: {trainer.config['epochs']}"

        # "## Model architecture\n" + f"{trainer.model.__str__}\n\n"
    )

    with open(os.path.join(trainer.logdir, "summary.txt"), 'w') as f:
        f.write(summary)

    if hasattr(trainer, "neptune_run"):
        trainer.neptune_run["summary"] = summary

def save_figures(trainer):
    for name, results in trainer.results.items():
        plt.plot([r["train"] for r in results], label="train")
        plt.plot([r["eval"] for r in results], label="eval")
        plt.title(name)
        plt.savefig(os.path.join(trainer.logdir, f"{name}-train-vs-eval.jpg"))
        plt.legend()
        plt.close()


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


def find_optimal_temperature(y_true, probs, temperatures_to_try=None):
    if temperatures_to_try is None:
        temperatures_to_try = np.linspace(0.1, 5.0, 50)
    
    best_temperature = 1.0
    best_metric = float('inf')
    metrics = []

    for temp in temperatures_to_try:
        scaled_probas = probs ** (1 / temp)
        
        scaled_probas = scaled_probas / scaled_probas.sum()
        
        log_loss_value = log_loss(y_true, scaled_probas)
        brier_score = brier_score_loss(y_true, scaled_probas)
        
        metrics.append({
            'temperature': temp,
            'log_loss': log_loss_value,
            'brier_score': brier_score
        })
        
        if log_loss_value < best_metric:
            best_metric = log_loss_value
            best_temperature = temp
    
    return best_temperature


def plot_probabilities(ax, probs, bins=100):
    ax.hist(probs, bins=bins, edgecolor='black', alpha=0.7)
    ax.grid(True)


def calibrate_and_plot(ax, y_true, probs_and_labels):
    for p, label in probs_and_labels:
        frac_pos, mean_probs = calibration_curve(y_true=y_true, y_prob=p, n_bins=10)
        ax.plot(mean_probs, frac_pos, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color='gray', label="perfectly calibrated")
    
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("frequency")
    ax.legend()
    ax.grid()

def plot_precision_recall_f1(ax, labels, probs):
    precision, recall, thresholds = precision_recall_curve(y_true=labels, y_score=probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    ax.plot(thresholds, precision[:-1], label="precision", color='blue')
    ax.plot(thresholds, recall[:-1], label="recall", color='green')
    ax.plot(thresholds, f1_scores[:-1], label="f1-score", color='red')
    
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Values")
    ax.legend()
    ax.grid()

def calibrate_and_plot_metrics(model, dataloader, device, train, writer=None, neptune_run=None):
    datatype = "train" if train else "eval"
    all_preds = []
    all_labels = []
    logits = []
    model.eval()
    
    for i, batch in enumerate(dataloader):
        batch = [a.to(device) for a in batch]
        with torch.no_grad():
            if model.__class__.__name__ == "MBHoloNet":
                output, _ = model(*batch[:-1])
            else:
                output = model(*batch[:-1])
        
        probs = torch.sigmoid(output).cpu().numpy()
        logits.extend(output.cpu().numpy().ravel())
        labels = batch[-1].cpu().numpy()
        
        all_preds.extend(probs.ravel())
        all_labels.extend(labels.ravel())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    logits = np.array(logits)
    

    iso_regressor = IsotonicRegression(out_of_bounds='clip')
    iso_calibrated_probs = iso_regressor.fit_transform(all_preds, all_labels)

    # temperature = find_optimal_temperature(y_true=all_labels, probs=all_preds, temperatures_to_try=[0.001, 0.01, 0.1, 1, 2, 3, 4, 5])
    temperature = 0.001
    temp_calibrated_probs = torch.sigmoid(torch.from_numpy(logits / temperature)).cpu().numpy()

    probs_and_labels = [(all_preds, "uncalibrated"), (iso_calibrated_probs, "iso-calibrated"), (temp_calibrated_probs, f"temp-calibrated-tem={temperature}")]

    calibrate_fig, ax = plt.subplots(figsize=(8,6))
    calibrate_and_plot(ax=ax, y_true=all_labels, probs_and_labels= probs_and_labels)
    plt.title(f"Calibration on {datatype} set")
    # plt.savefig(f"calibration_probabilities-{datatype}.png")

    plot_metrics_fig, axes = plt.subplots(2, 3, figsize=(19.2,10.8))
    for i, (p, label) in enumerate(probs_and_labels):
        plot_probabilities(ax=axes[0, i], probs=all_preds)
        plot_precision_recall_f1(ax=axes[1, i], labels=all_labels, probs=p)
        axes[0,i].set_title(label)
        axes[1,i].set_title(label)
    plt.suptitle(f"Metrics as a function of threshold on {datatype} set")
    # plt.savefig(f"precision_recall_curve-{datatype}.png")

    
    if writer:
        writer.add_figure(f"calibration_probabilities/{datatype}", calibrate_fig)
        writer.add_figure(f"precision_recall_curve/{datatype}", plot_metrics_fig)
    if neptune_run:
        neptune_run[f"{datatype}/calibration_probabilities"].upload(calibrate_fig)
        neptune_run[f"{datatype}/precision_recall_curve"].upload(plot_metrics_fig)

