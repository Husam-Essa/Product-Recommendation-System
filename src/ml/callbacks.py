from tensorflow.keras.callbacks import Callback

class DelayedEarlyStopping(Callback):
    def __init__(self, start_epoch=50, patience=10, monitor="val_auc", mode="max", restore_best_weights=True, verbose=1):
        super().__init__()
        self.start_epoch = start_epoch
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.best = -float("inf") if mode == "max" else float("inf")
        self.best_weights = None
        self.stopped_epoch = 0
        self.verbose = verbose
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if epoch < self.start_epoch:
            return
        current = logs.get(self.monitor)
        if current is None:
            return
        improved = (self.mode == "max" and current > self.best) or (self.mode == "min" and current < self.best)
        if improved:
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                if self.verbose:
                    print(f"\nEarly stopping at epoch {epoch+1} (no improvement after {self.patience} epochs).")
