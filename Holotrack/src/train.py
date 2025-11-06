import os
import yaml
import torch
import neptune

# Local imports
from src import models
from src import data
from src import utils
from src import schedulers
from src import metrics
from src import losses
from src import callbacks

class Trainer:
    def __init__(self, config, on_train=True):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.config = config
        self.on_train = on_train

    def train_settings(self):
        logdir = utils.get_logdir(self.config["logdir"])
        os.makedirs("logs", exist_ok=True)
        os.makedirs(logdir, exist_ok=True)

        with open("configs/logs.yaml", "r") as f:
            log_config_dict = yaml.safe_load(f)
        log_config_dict["handlers"]["file"]["filename"] = os.path.join(logdir, "training.log")
        logging.config.dictConfig(log_config_dict)
        self.logger = logging.getLogger("train")
        self.logger.info(f"Training on {self.device}")

        self.logger.info(f"Logdir: {logdir}")
        with open(os.path.join(logdir, "config.yaml"), 'w') as f:
            yaml.safe_dump(self.config, f)

        self.logdir = logdir

        seed = self.config.get("seed")
        # if seed is not None: utils.set_seed(seed) # TODO: Uncomment for reproductibilty

        self.writer = SummaryWriter(log_dir=logdir)

        if "neptune" in self.config:
            self.neptune_run = neptune.init_run(project=self.config["neptune"])
            self.logger.info(f"Neptune: {self.config['neptune']}")
