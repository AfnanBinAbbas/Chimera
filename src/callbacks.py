from __future__ import annotations
from pytorch_lightning.callbacks import Callback
from src.metrics_eval import compute_ad_rca_metrics
from src.dataset import get_loader


class ValidationMetricsCallback(Callback):
    """Compute AD+RCA metrics on the validation split each epoch and log AD F1.

    The callback builds small evaluation loaders from the provided file paths
    using `get_eval_loader` and calls `compute_ad_rca_metrics` to get AD F1.
    It logs `val_ad_f1` (higher is better) so `ModelCheckpoint` and
    `EarlyStopping` can monitor it.
    """

    def __init__(self, n_val_fp: str, an_val_fp: str, batch_size: int = 128, device=None):
        self.n_val_fp = str(n_val_fp)
        self.an_val_fp = str(an_val_fp)
        self.batch_size = int(batch_size)
        self.device = device

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        try:
            # Build validation loaders using the same paired structure as main evaluation.
            n_loader = get_loader(self.n_val_fp, self.n_val_fp, batch_size=self.batch_size, shuffle=False, num_workers=0)
            an_loader = get_loader(self.an_val_fp, self.an_val_fp, batch_size=self.batch_size, shuffle=False, num_workers=0)

            # Ensure model in eval mode and on correct device
            pl_module.eval()

            metrics = compute_ad_rca_metrics(pl_module, n_loader, an_loader)
            val_f1 = float(metrics.get("ad_f1", 0.0))
            # Log to Lightning so callbacks can monitor it
            pl_module.log("val_ad_f1", val_f1, on_epoch=True, prog_bar=True, logger=True)
            print(f"[ValidationMetricsCallback] epoch={trainer.current_epoch} val_ad_f1={val_f1:.4f}")
        except Exception as exc:
            print(f"ValidationMetricsCallback failed: {exc}")
