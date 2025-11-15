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

    def prepare_data(self):
        klass_name, params = self.get_config_item(name="data")
        klass = getattr(data, klass_name, None) # data package
        assert klass is not None, f"Class {klass_name} is not defined in data"
        self.data = klass(**params)
        self.data.split()
        self.train_dataloader = self.data.train_dataloader()
        self.eval_dataloader = self.data.eval_dataloader()

        if self.on_train:
            self.logger.info(f"Dataset: {self.data}")
            self.config["data"]["params"] = {**self.config["data"]["params"], **self.data.params}
            self.logger.info(f"Train dataset size: {len(self.train_dataloader.dataset)}")
            self.logger.info(f"Eval dataset size: {len(self.eval_dataloader.dataset)}")
            self.logger.info(f"Train indices: {self.data.train_indices}")
            self.logger.info(f"Eval indices: {self.data.valid_indices}")

    def prepare_model(self):
        klass_name, params = self.get_config_item(name="loss")
        klass = getattr(nn, klass_name, None) # nn package
        if klass is None:
            klass = getattr(losses, klass_name, None) # nn package
        assert klass is not None, f"Class {klass_name} is not defined in nn"
        self.loss = klass(**params)

        klass_name, params = self.get_config_item(name="model")
        klass = getattr(models, klass_name, None) # models package
        assert klass is not None, f"Class {klass_name} is not defined in models"
        self.model = klass(**params).to(self.device)

        if self.on_train:
            self.logger.info(f"Loss: {self.loss}")
            self.logger.info(f"Model:\n{self.model}")

            model_chkpnt_save = os.path.join(self.logdir, "best_model.pt")
            self.model_checkpoint = utils.ModelCheckpoint(self.model, model_chkpnt_save, min_is_best=True)
            self.logger.info(f"Model checkpoint: {model_chkpnt_save}")


    def configure_optimizers(self):
        klass_name, params = self.get_config_item(name="optimizer")
        params = {
            "params": self.model.parameters(),
            **params
        }
        klass = getattr(optimizers, klass_name, None) # optimizers package
        assert klass is not None, f"Class {klass_name} is not defined in optimizers"
        self.optimizer= klass(**params)
        self.logger.info(f"Optimizer: {self.optimizer}")
        klass_name, params = self.get_config_item(name="lr_scheduler")
        if klass_name is not None:
            params = {
                "optimizer": self.optimizer,
                **params
            }
            klass = getattr(torch.optim.lr_scheduler, klass_name, None) # optimizers package
            if klass is None:
                klass = getattr(schedulers, klass_name, None) # optimizers package
            self.lr_scheduler = klass(**params)
            self.logger.info(f"Learning rate scheduler: {self.lr_scheduler}")

    def configure_metrics(self):
        names = self.config.get("metrics")
        self.metric_funcs = {n: getattr(metrics, n, None) for n in names}
        self.compute_metrics = lambda y_true, y_pred: {n: self.metric_funcs[n](y_true=y_true, y_pred=y_pred).item() for n in self.metric_funcs}

        if self.on_train:
            self.logger.info(f"Metrics: {self.metric_funcs}")

    def configure_callbacks(self):
        self.cm_callbacks = [callbacks.BinaryMetricsCallback(trainer=self, num_classes=2, threshold=th, add_cm=False)
                            for th in [0.2, 0.4, 0.6, 0.8] ] + 
                            [callbacks.BinaryMetricsCallback(trainer=self, num_classes=2, threshold=0.5, add_cm=False)]
        self.cm_callbacks = [callbacks.BinaryMetricsCallback(trainer=self, num_classes=2, threshold=0.5, add_cm=True)]
