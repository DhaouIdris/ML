import yaml

with open("Config.yaml", "r") as f:
    config = yaml.safe_load(f)


def get_optimizer(config, params):
    params_dict = config["optim"]["params"]
    algo = config["optim"]["algo"]
    optimizer = eval(f"torch.optim.{algo}(params, **params_dict)")
    return optimizer
