import os
import yaml
from argparse import ArgumentParser

root = "/usr/users/sdim/%s/%s/data/%s"
save_dir="/usr/users/sdim/%s/%s/data"
logdir = "/usr/users/sdim/%s/%s/logs/mb-holonet"
neptune="%s/holotrack"

# neptune_user = "bmalick"
# user="sdim_21"
# pwd="holotrack"

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--user", "-u", required=True, type=str, help="sdi user account. ex: sdim_21")
    parser.add_argument("--pwd", "-p", required=True, type=str, help="project pwd into ~ in dce. ex: holotrack, Documents/holotrack")
    parser.add_argument("--neptune", "-n", required=True, type=str, help="neptune user name. ex: bmalick")

    args = parser.parse_args()
    user = args.user
    pwd = args.pwd
    neptune_user = args.neptune

    os.makedirs("myconfigs", exist_ok=True)

    configs = [entry for entry in os.scandir("configs/dce")]
    for entry in configs:
        with open(entry.path, 'r') as f:
            data = yaml.safe_load(f)
        if "save_dir" in data:
            data["save_dir"] = save_dir % (user, pwd)
        if "data" in data:
            data_name = data["data"]["params"]["root"].split("/")[-1]
            data["data"]["params"]["root"] = root % (user, pwd, data_name)
        if "logdir" in data:
            data["logdir"] = logdir % (user, pwd)
        if "neptune" in data:
            data["neptune"] = neptune % neptune_user
        fname = os.path.join("myconfigs", entry.name)
        with open(fname, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)
        print(f"New train config saved into {fname}")
