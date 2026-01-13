"""
Early Stopping callback for training.
"""


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.

    Args:
        patience: Number of epochs with no improvement after which training stops
        mode: 'min' for loss, 'max' for accuracy
        delta: Minimum change to qualify as an improvement
        verbose: If True, prints a message for each improvement
    """

    def __init__(self, patience=5, mode="max", delta=0.0, verbose=True):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_best = None

        if mode == "min":
            self.monitor_op = lambda x, best: x < best - delta
        else:  # mode == 'max'
            self.monitor_op = lambda x, best: x > best + delta

    def __call__(self, val_metric, model=None, save_path=None):
        """
        Check if should stop training.

        Args:
            val_metric: Current validation metric value
            model: Model to save if improved (optional)
            save_path: Path to save checkpoint (optional)

        Returns:
            True if should stop training
        """
        score = val_metric if self.mode == "max" else -val_metric

        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.val_best = val_metric
            if model is not None and save_path is not None:
                self._save_checkpoint(model, save_path, val_metric)
        elif self.monitor_op(val_metric, self.val_best):
            # Improvement
            self.best_score = score
            self.val_best = val_metric
            self.counter = 0
            if model is not None and save_path is not None:
                self._save_checkpoint(model, save_path, val_metric)
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _save_checkpoint(self, model, save_path, val_metric):
        """Save model checkpoint."""
        import torch
        from pathlib import Path

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), save_path)

        if self.verbose:
            print(
                f"Validation metric improved ({self.val_best:.4f} → {val_metric:.4f}). Saving model to {save_path}..."
            )


if __name__ == "__main__":
    # Test early stopping
    import torch
    import torch.nn as nn

    print("Testing Early Stopping...")

    # Create dummy model
    model = nn.Linear(10, 1)

    # Create early stopping
    early_stopping = EarlyStopping(patience=3, mode="max", verbose=True)

    # Simulate training
    val_accs = [0.5, 0.55, 0.56, 0.55, 0.54, 0.53, 0.52]  # Peak at epoch 3

    for epoch, val_acc in enumerate(val_accs, 1):
        print(f"\nEpoch {epoch}, Val Acc: {val_acc:.2f}")

        should_stop = early_stopping(val_acc)

        if should_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"Best val acc: {early_stopping.val_best:.2f}")
            break
