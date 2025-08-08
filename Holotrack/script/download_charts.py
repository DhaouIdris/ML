import neptune
import matplotlib.pyplot as plt
import argparse
import numpy as np

METRICS = [
    "Loss", "Mae", "Pcc", "Psnr", "R2_Score",
    "MetricCallbacks_th=0.5-F1_Score",
    "MetricCallbacks_th=0.5-Precision",
    "MetricCallbacks_th=0.5-Recall"
]


def plot_train_vs_eval(run, metric_name, num_ticks=4, margin_factor=0.05):
    """Plot train vs eval metric and return the figure."""
    train_series = run[f"train/{metric_name}"].fetch_values()
    eval_series = run[f"eval/{metric_name}"].fetch_values()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(train_series["timestamp"], train_series["value"], label="train")
    ax.plot(eval_series["timestamp"], eval_series["value"], label="eval")

    xticks = train_series["timestamp"]
    xtick_labels = train_series["step"]
    tick_indices = np.linspace(0, len(xticks) - 1, num_ticks, dtype=int)
    ax.set_xticks(xticks[tick_indices])
    ax.set_xticklabels([f'{int(label)}' for label in xtick_labels[tick_indices]])

    y_min = min(train_series["value"].min(), eval_series["value"].min())
    y_max = max(train_series["value"].max(), eval_series["value"].max())
    y_ticks = np.linspace(y_min, y_max, num_ticks)
    ax.set_yticks(y_ticks)

    x_margin = (xticks.max() - xticks.min()) * margin_factor
    y_margin = (y_max - y_min) * margin_factor

    ax.set_xlim([xticks.min() - x_margin, xticks.max() + x_margin])
    ax.set_ylim([y_min - y_margin, y_max + y_margin])

    ax.set_xlabel("Steps")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} - train vs eval")
    ax.legend()

    ax.grid(True, color='gray', linestyle='--', linewidth=0.3)
    path = f"report/images/{run['sys/id'].fetch()}-{metric_name}-train-vs-eval.png"
    plt.savefig(path)
    plt.close()
    print(metric_name, f": saving figure to {path}")

def compare_runs(runs, metrics):
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    if len(runs)>=5:
        print("Max 5 runs")
        return

    for set_name in ["train", "eval"]:
        for metric_name in metrics:
            fig, ax = plt.subplots(figsize=(10, 8))
            for i,run in enumerate(runs):
                series = run[f"{set_name}/{metric_name}"].fetch_values()
                ax.plot(series["step"], series["value"], label=run['sys/id'].fetch(), color=colors[i])

            ax.set_xlabel("Steps")
            ax.set_ylabel(metric_name)
            ax.set_title(f"{set_name}/{metric_name}")
            ax.legend()

            ax.grid(True, color='gray', linestyle='--', linewidth=0.3)

            run_names = "-".join([run['sys/id'].fetch().split("-")[-1] for run in runs])
            path = f"report/images/Hols-{run_names}-{set_name}-{metric_name}.png"
            plt.savefig(path)
            plt.close()
            print(metric_name, f": saving figure to {path}")

def main():
    parser = argparse.ArgumentParser(description="Download and plot train vs eval metrics from Neptune")
    parser.add_argument("--project", "-p", required=True, type=str, help="Neptune project name (e.g., user/project)")
    parser.add_argument("--run_ids", "-r", required=True, type=str, help="Neptune run ID (e.g., HOL-80)")
    parser.add_argument("--compare-runs", action="store_true", help="Compare runs")

    args = parser.parse_args()

    if not args.compare_runs:
        # Connect to the run
        run = neptune.init_run(
            project=args.project,
            with_id=args.run_ids.upper(),
        )

        print(f"Generating report for run {args.run_ids.upper()}...")
        print()
        for name in METRICS:
            plot_train_vs_eval(run=run, metric_name=name)
    else:
        runs = [neptune.init_run(project=args.project, with_id=run_id.upper())
                for run_id in args.run_ids.split()
        ]
        compare_runs(runs, METRICS)

if __name__ == "__main__":
    main()
