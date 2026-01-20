import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.profiler import profile, ProfilerActivity
import logging

from .model import Model
from .data import get_datasets

# --- M14: Advanced Logging Setup ---
log_dir = Path("logs_logging")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,  # Using INFO for training to keep the console clean
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_dir / "training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs"
_CONFIG_PATH_STR = str(_CONFIG_PATH.resolve())
if not _CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Config directory not found at {_CONFIG_PATH_STR}. Current file: {__file__}"
    )


@hydra.main(version_base=None, config_path=_CONFIG_PATH_STR, config_name="config")
def train(cfg: DictConfig):
    """
    Train the translation model using Hydra configuration.

    Args:
        cfg: Hydra configuration object containing all hyperparameters
    """
    # Extract configuration values
    processed_data_dir = cfg.paths.processed_data_dir
    checkpoint_dir = cfg.paths.checkpoint_dir

    batch_size = cfg.train.batch_size
    epochs = cfg.train.epochs
    # lr = cfg.model.lr

    print("=" * 50)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 50)

    logger.info("Loading datasets...")
    train_set, eval_set, _ = get_datasets(processed_dir=processed_data_dir)

    # Use subsets to keep runtime reasonable
    train_set = Subset(
        train_set, range(min(len(train_set), cfg.train.train_subset_size))
    )
    eval_set = Subset(eval_set, range(min(len(eval_set), cfg.train.eval_subset_size)))
    print(f"DEBUG: Running with {len(train_set)} training samples")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=cfg.train.shuffle,
        num_workers=cfg.train.num_workers,
    )
    val_loader = DataLoader(
        eval_set,
        batch_size=batch_size,
        num_workers=cfg.train.num_workers,
    )

    print("Initializing model")
    model = Model(
        lr=cfg.model.lr,
        batch_size=cfg.model.batch_size,
        max_source_length=cfg.model.max_source_length,
        max_target_length=cfg.model.max_target_length,
        model_name=cfg.model.model_name,
        prefix=cfg.model.prefix,
        max_new_tokens=cfg.model.max_new_tokens,
        num_beams=cfg.model.num_beams,
    )

    # --------------------------------------------------
    # 🔍 PROFILING BLOCK (M13)
    # --------------------------------------------------
    if cfg.train.enable_profiling:
        print("\nRunning PyTorch profiler on one training step")

        model.train()
        batch = next(iter(train_loader))

        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
        ) as prof:
            model.training_step(batch, batch_idx=0)

        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # --------------------------------------------------
    # Normal training
    # --------------------------------------------------
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=cfg.train.checkpoint_filename,
        monitor=cfg.train.monitor,
        mode=cfg.train.mode,
        save_top_k=cfg.train.save_top_k,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        callbacks=[checkpoint_callback],
        log_every_n_steps=cfg.train.log_every_n_steps,
    )

    logger.info("Starting trainer.fit()...")
    trainer.fit(model, train_loader, val_loader)

    logger.info("Training complete. Checkpoints saved to: %s", checkpoint_dir)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.exception(f"Training process failed! {e}")
        raise
