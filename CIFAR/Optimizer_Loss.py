import yaml

with open("Config.yaml", "r") as f:
    config = yaml.safe_load(f)


def get_optimizer(config, params):
    params_dict = config["optim"]["params"]
    algo = config["optim"]["algo"]
    optimizer = eval(f"torch.optim.{algo}(params, **params_dict)")
    return optimizer

def loss_function(config):

    loss_fct = config["loss"]
    loss = eval(f"nn.{loss_fct}()")
    return loss
