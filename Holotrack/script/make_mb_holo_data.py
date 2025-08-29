import yaml
from argparse import ArgumentParser
from src.data import mb_holonet

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file", "-f", type=str, required=True, help="config file containing mbholo params for holograms creation")
    args = parser.parse_args()
    with open(args.file, "r") as f:
        params = yaml.safe_load(f)
    params = mb_holonet.MBHolonetParams(**params)
    print(f"Creating holograms using {args.file}")
    mb_holonet.MBHolonetHologram(params).make_holograms()
