import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.profiler import profile, ProfilerActivity

# --- M14: W&B Imports ---
from pytorch_lightning.loggers import WandbLogger
import wandb
import logging

from .model import Model
from .data import get_datasets

# Logging Setup
log_dir = Path("logs_logging")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "training.log"),
        logging.StreamHandler()
    ]
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
    # --- M14: SWEEP INTEGRATION ---
    # 1. Initialize W&B. If this is a sweep, wandb.init() picks up the sweep params.
    run = wandb.init() 
    
    # 2. Override Hydra config with Sweep parameters
    # This ensures that cfg.model.lr becomes the value chosen by the sweep.
    if wandb.run:
        for key, value in wandb.config.items():
            logger.info(f"Sweep override: {key} = {value}")
            # This handles nested keys if they exist (e.g., lr)
            OmegaConf.update(cfg, key, value, merge=True)

    # Now we use cfg normally, and it contains the sweep values
    processed_data_dir = cfg.paths.processed_data_dir
    checkpoint_dir = cfg.paths.checkpoint_dir
    
    # Use WandbLogger and pass the resolved config
    wandb_logger = WandbLogger(
        project="en-es-translation",
        config=OmegaConf.to_container(cfg, resolve=True),
        log_model="all"
    )

    logger.info("Loading datasets...")
    train_set, eval_set, _ = get_datasets(processed_dir=processed_data_dir)

    train_set = Subset(train_set, range(min(len(train_set), cfg.train.train_subset_size)))
    eval_set = Subset(eval_set, range(min(len(eval_set), cfg.train.eval_subset_size)))

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train.batch_size, # This will be the sweep value (16 or 32)
        shuffle=cfg.train.shuffle,
        num_workers=cfg.train.num_workers,
    )
    val_loader = DataLoader(
        eval_set,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
    )

    model = Model(
        lr=cfg.model.lr, # This will be the sweep value (1e-4, 5e-5, or 1e-5)
        batch_size=cfg.train.batch_size,
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
        monitor="val_loss", # Must match the 'metric' in sweep.yaml
        mode="min",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=cfg.train.log_every_n_steps,
    )

    trainer.fit(model, train_loader, val_loader)
    wandb.finish()

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.exception(f"Training process failed! {e}")
        raise
