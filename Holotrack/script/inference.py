import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from argparse import ArgumentParser
from matplotlib.animation import FFMpegWriter
from torchvision import transforms


from src import train

if __name__ == "__main__":
    os.makedirs("inference", exist_ok=True)

    parser = ArgumentParser()
    parser.add_argument("--log", "-l", required=True, type=str, help="Logs folder. Ex: logs/mb-holonet--2025-03-18--17-16-19")
    parser.add_argument("--indice", "-i", default=None, type=int, help="Dataset indice")
    parser.add_argument("--model", "-m", default="best_model", type=str, help="Model checkpoint")
    parser.add_argument("--fps", default=1.5, type=int, help="Frames per second for the animation")
    parser.add_argument("--crop", action="store_true", help="enable cropping in hologram: drop 5% of the hologram")
    args = parser.parse_args()

    log = args.log
    save_dir = os.path.join("inference", log.split("/")[-1])
    os.makedirs(save_dir, exist_ok=True)
    print(f"Logs: {log}")

    config = yaml.safe_load(open(f"{log}/config.yaml", "r"))
    trainer = train.Trainer(config, on_train=False)
    trainer.prepare_model()
    trainer.prepare_data()
    model = trainer.model
    model.eval()
    model.load_state_dict(torch.load(f"{log}/{args.model}.pt"))

    data = trainer.data

    with open(data.otf_fname, "rb") as f: 
        otf3d = np.load(f)
    otf3d = torch.from_numpy(otf3d[np.newaxis]).to(trainer.device)

    idx = np.random.randint(len(data)) if args.indice is None else args.indice
    save_dir = os.path.join(save_dir, f"item-{idx}"+("-cropped" if args.crop else ""))
    os.makedirs(save_dir, exist_ok=True)
    img, ground_truth = data[idx]

    if args.crop:
        crop_shape = int(0.95 * img.shape[0])
        transform = transforms.Compose([
            transforms.RandomCrop(crop_shape),
            transforms.Resize(img.shape[0]),
        ])
        img = transform(torch.from_numpy(img[np.newaxis])).numpy()[0]

    temperature = 1
    threshold = 0.5

    ex = torch.from_numpy(img[np.newaxis]).to(trainer.device)
    with torch.no_grad():
        pred_volume, _, hidden_states = model(ex, otf3d, return_vols=True)
    pred_volume /= temperature
    probas = torch.sigmoid(pred_volume)
    pred_volume = (probas > threshold).int().cpu().detach().numpy()[0]

    gt_color = 'red'
    pred_color = 'blue'

    fig = plt.figure(figsize=(14, 6))

    ax_img = fig.add_subplot(1, 2, 1)
    ax_img.imshow(img, cmap='gray')
    ax_img.set_title("hologram")
    ax_img.axis("off")

    ax_3d = fig.add_subplot(1, 2, 2, projection='3d')

    data.plot_3d(positions=ground_truth, ax=ax_3d, color=gt_color, label="gt", marker='o')
    v0_probas = torch.sigmoid(hidden_states[0] / temperature)
    v0_pred = (v0_probas > threshold).int().detach().cpu().numpy()[0]
    pred_points = data.plot_3d(v0_pred, ax=ax_3d, color=pred_color, label="pred", marker='x')
    
    title = ax_3d.set_title(f"Stage 0")
    ax_3d.legend()
    
    all_preds = []
    for i, v in enumerate(hidden_states):
        v_probas = torch.sigmoid(v / temperature)
        v_pred = (v_probas > threshold).int().detach().cpu().numpy()[0]
        all_preds.append(v_pred)
        
        fig_frame, ax_frame = plt.subplots(1, 1, figsize=(12, 6), subplot_kw={'projection': '3d'})
        data.plot_3d(positions=ground_truth, ax=ax_frame, color=gt_color, label="gt", marker='o')
        data.plot_3d(v_pred, ax=ax_frame, color=pred_color, label="pred", marker='x')
        plt.title(f"Stage {i}")
        ax_frame.legend()
        plt.savefig(f"{save_dir}/stage-{i}.png")
        plt.close(fig_frame)
    
    def update(frame):
        ax_3d.collections[-1].remove()
        data.plot_3d(all_preds[frame], ax=ax_3d, color=pred_color, label="pred", marker='x')
        title.set_text(f"Stage {frame}")
        return ax_3d,
    
    ani = animation.FuncAnimation(
        fig, update, frames=len(all_preds), 
        interval=1000/args.fps, blit=False
    )
    
    writer = FFMpegWriter(fps=args.fps)
    ani.save(f"{save_dir}/animation.gif", writer=writer)
    print(f"Animation saved to {save_dir}/animation.gif")
    plt.show()
    
    print(f"Save dir: {save_dir}")

