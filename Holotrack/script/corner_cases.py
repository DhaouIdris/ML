import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from argparse import ArgumentParser
from matplotlib.animation import FFMpegWriter

from src import train
from src.data import mb_holonet



def make_hologram(data_creator, volume: np.ndarray):
    otf3d = data_creator.kernel_projection(data_creator.params.pps, data_creator.params.wavelength,
                                    data_creator.params.z_range, data_creator.params.nxy, data_creator.params.nxy)

    volume_prime = 1-volume
    hologram = - data_creator.gabor_hologram(volume_prime, otf3d, data_creator.params.noise_level)
    return hologram

empty_volume = lambda : np.zeros((data_creator.params.nxy, data_creator.params.nxy, data_creator.params.nz))

def particle_in_corners(pos=[0,0,0]):
    volume = empty_volume()
    x,y,z = pos
    volume[x, y, z] = 1
    return volume

def random_corners(num_corners: int=1):
    assert 0 < num_corners < 9, "1 <= num_corners <= 8"
    volume = empty_volume()
    corners = ([0,0,0], [0,0,data_creator.params.nz-1], [0,data_creator.params.nxy-1,0],
               [0,data_creator.params.nxy-1,data_creator.params.nz-1], [data_creator.params.nxy-1,0,0],
               [data_creator.params.nxy-1,0,data_creator.params.nz-1],
               [data_creator.params.nxy-1,data_creator.params.nxy-1,0],
               [data_creator.params.nxy-1,data_creator.params.nxy-1,data_creator.params.nz-1])
    rnd_corners = [corners[i] for i in np.random.randint(0, len(corners), num_corners)]
    for x, y, z in rnd_corners: volume[x, y, z] = 1
    return volume

def centered_particle():
    volume = empty_volume()
    x, y, z = (data_creator.params.nxy//2 , data_creator.params.nxy//2 , data_creator.params.nz//2)
    volume[x, y, z] = 1
    return volume

def inline_particles(num_particles=2):
    volume = empty_volume()
    volume[data_creator.params.nxy//2-5:data_creator.params.nxy//2-5+num_particles , data_creator.params.nxy//2 , data_creator.params.nz//2] = 1
    return volume

def predict(volume, name, fps=1.5):
    temperature = 1
    threshold = 0.5
    gt_color = 'red'
    pred_color = 'blue'

    os.makedirs("corners_cases", exist_ok=True)
    save_dir = f"corners_cases/{name}"
    os.makedirs(save_dir, exist_ok=True)

    hologram = make_hologram(data_creator=data_creator, volume=volume)
    ex = torch.from_numpy(hologram[np.newaxis]).to(trainer.device)
    with torch.no_grad():
        pred_volume, _, hidden_states = model(ex, otf3d, return_vols=True)
    pred_volume /= temperature
    probas = torch.sigmoid(pred_volume)
    pred_volume = (probas > threshold).int().cpu().detach().numpy()[0]

    fig = plt.figure(figsize=(14, 6))
    plt.suptitle(name)

    ax_img = fig.add_subplot(1, 2, 1)
    ax_img.imshow(hologram, cmap='gray')
    ax_img.set_title("hologram")
    ax_img.axis("off")

    ax_3d = fig.add_subplot(1, 2, 2, projection='3d')

    data.plot_3d(positions=volume, ax=ax_3d, color=gt_color, label="gt", marker='o')
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
        data.plot_3d(positions=volume, ax=ax_frame, color=gt_color, label="gt", marker='o')
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
    # plt.show()
    
    fig.savefig(f"{save_dir}/stage-{5}-fig.png")
    print(f"Save dir: {save_dir}")



if __name__ == "__main__":
    os.makedirs("inference", exist_ok=True)

    parser = ArgumentParser()
    parser.add_argument("--log", "-l", required=True, type=str, help="Logs folder. Ex: logs/mb-holonet--2025-03-18--17-16-19")
    # parser.add_argument("--indice", "-i", default=None, type=int, help="Dataset indice")
    parser.add_argument("--model", "-m", default="best_model", type=str, help="Model checkpoint")
    parser.add_argument("--fps", default=1.5, type=int, help="Frames per second for the animation")
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
    data_config = {
        "save_dir": None,
        "name": None,
        "dtype": "float64",
        "save_one_npy": False,
        "sr": 0,  # pixel size of half random point
        "data_num": None,  # number of train data_single
        "holo_data_type": 1,
        "nxy": 32,
        "nz": 7,
        "dz": 1.2e-3,  # depth interval of the object slices
        "object_type": "sim",
        "wavelength": 660e-9,  # Illumination wavelength
        "pps": 20e-6,  # pixel pitch of CCD camera
        "z0": 5e-3,  # Distance between the hologram and the center plane of the 3D object
        "noise_level": 50,  # DB of the noise
        "group_num": 1
    }

    params = mb_holonet.MBHolonetParams(**data_config)
    data_creator = mb_holonet.MBHolonetHologram(params)

    predict(volume=centered_particle(), name="centered-particle", fps=args.fps)

    predict(volume=particle_in_corners(pos=[0,0,0]), name="particle-in-corner-0-0-0", fps=args.fps)
    predict(volume=particle_in_corners(pos=[0,0,data_creator.params.nz-1]), name=f"particle-in-corner-0-0-{data_creator.params.nz}", fps=args.fps)
    predict(volume=particle_in_corners(pos=[0,data_creator.params.nxy-1,0]), name=f"particle-in-corner-0-{data_creator.params.nxy}-0", fps=args.fps)
    predict(volume=particle_in_corners(pos=[0,data_creator.params.nxy-1,data_creator.params.nz-1]), name=f"particle-in-corner-0-{data_creator.params.nxy}-{data_creator.params.nz}", fps=args.fps)
    predict(volume=particle_in_corners(pos=[data_creator.params.nxy-1,0,0]), name=f"particle-in-corner-{data_creator.params.nxy}-0-0", fps=args.fps)
    predict(volume=particle_in_corners(pos=[data_creator.params.nxy-1,0,data_creator.params.nz-1]), name=f"particle-in-corner-{data_creator.params.nxy}-0-{data_creator.params.nz}", fps=args.fps)
    predict(volume=particle_in_corners(pos=[data_creator.params.nxy-1,data_creator.params.nxy-1,0]), name=f"particle-in-corner-{data_creator.params.nxy}-{data_creator.params.nxy}-0", fps=args.fps)
    predict(volume=particle_in_corners(pos=[data_creator.params.nxy-1,data_creator.params.nxy-1,data_creator.params.nz-1]), name=f"particle-in-corner-{data_creator.params.nxy}-{data_creator.params.nxy}-{data_creator.params.nz}", fps=args.fps)

    predict(volume=random_corners(num_corners=2), name="random-corners-2", fps=args.fps)
    predict(volume=random_corners(num_corners=4), name="random-corners-4", fps=args.fps)

    predict(volume=inline_particles(num_particles=2), name="inline-particles-two", fps=args.fps)
    predict(volume=inline_particles(num_particles=3), name="inline-particles-three", fps=args.fps)
    predict(volume=inline_particles(num_particles=4), name="inline-particles-four", fps=args.fps)
