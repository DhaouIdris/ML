class Callback:
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_start(self):
        self.all_preds = []
        self.all_labels = []

    def on_epoch_end(self):
        pass

    def on_batch_end(self, train: bool, **kwargs):
        pass

