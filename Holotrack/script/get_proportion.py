from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
from src import data

def main():
    parser = ArgumentParser()
    parser.add_argument("--folder", "-f", type=str, required=True, help="Data directory")
    parser.add_argument("--data_type", "-d", type=str, required=True, default="bacteria", help="Dataset type: bacteria or mbholonet")
    args = parser.parse_args()

    if args.data_type=="mbholonet":
        dataset = data.MBHoloDataset(root=args.folder)
    elif args.data_type=="bacteria":
        dataset = data.BacteriaDataset(root=args.folder)
    else:
        print(f"{args.data_type} is unknown. Use bacteria or mbholonet")
    proportions = []
    for _,p in tqdm(dataset):
        proportions.append([p.sum(), np.prod(p.shape)])
    proportions = np.array(proportions)
    proportions = proportions.sum(axis=0)
    num_particles, total = proportions
    print("Part of 1s:", num_particles/total)
    print("Part of 0s:", 1-num_particles/total)


if __name__=="__main__": main()
