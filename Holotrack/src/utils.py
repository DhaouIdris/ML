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