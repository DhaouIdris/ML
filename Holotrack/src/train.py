import os
import yaml
import torch
import neptune
import logging
import logging.config

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


    def get_config_item(self, name: str):
        args = self.config.get(name)
        if args is None: return None, None
        assert args is not None, f"{name} does not appear in config file"
        name = args.get("class")
        params = args.get("params", {})
        return name, params

    def fit(self):
        self.train_settings()
        self.prepare_data()
        self.prepare_model()
        self.configure_optimizers()
        self.configure_metrics()
        self.configure_callbacks()
        self.logs_tracker = callbacks.LogsTracker(trainer=self)

        if "neptune" in self.config:
            neptune_config = {k:v if not isinstance(v, list) else "[" + ",".join(str(v)) + "]" for k,v in self.config.items()}
            self.neptune_run["config"] = neptune_config

        utils.get_summary(self)
        self.results = {"loss": []}

        self.epoch = 0
        self.logger.info("Start of training")
        for _ in range(self.config["epochs"]):
            self.fit_epoch()
            self.epoch += 1

        utils.calibrate_and_plot_metrics(model=self.model, dataloader=self.train_dataloader, train=True,
                                 device=self.device, writer=self.writer, neptune_run=getattr(self, "neptune_run", None))
        utils.calibrate_and_plot_metrics(model=self.model, dataloader=self.eval_dataloader, train=False,
                                 device=self.device, writer=self.writer, neptune_run=getattr(self, "neptune_run", None))


        torch.save(self.model.state_dict(), os.path.join(self.logdir, "last-checkpoint.pt"))
        self.logger.info("End of training")

        if "neptune" in self.config:
            self.neptune_run["logs/training"].upload(os.path.join(self.logdir, "training.log"))
            self.neptune_run.stop()
