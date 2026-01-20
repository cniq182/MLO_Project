import os
import logging
import torch
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.profiler import profile, ProfilerActivity

from model import Model
from data import get_datasets

# --- M14: Advanced Logging Setup ---
log_dir = Path("logs_logging")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO, # Using INFO for training to keep the console clean
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train():
    processed_data_dir = "en_es_translation/data/processed"
    checkpoint_dir = "en_es_translation/models/checkpoints"

    batch_size = 8
    epochs = 1
    lr = 1e-3

    logger.info("Loading datasets...")
    train_set, eval_set, _ = get_datasets(processed_dir=processed_data_dir)

    # Use subsets to keep runtime reasonable
    train_size = min(len(train_set), 500)
    eval_size = min(len(eval_set), 100)
    train_set = Subset(train_set, range(train_size))
    eval_set = Subset(eval_set, range(eval_size))
    
    logger.info(f"DEBUG: Running with {train_size} training samples and {eval_size} eval samples.")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        eval_set,
        batch_size=batch_size,
        num_workers=0,
    )

    logger.info(f"Initializing model with learning rate: {lr}")
    model = Model(lr=lr, batch_size=batch_size)

    # --------------------------------------------------
    # 🔍 PROFILING BLOCK (M13)
    # --------------------------------------------------
    logger.info("Running PyTorch profiler on one training step...")

    model.train()
    batch = next(iter(train_loader))

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof:
        model.training_step(batch, batch_idx=0)

    # Logging the profiler table instead of just printing it
    logger.info("\n" + prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # --------------------------------------------------
    # Normal training
    # --------------------------------------------------
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
    )

    logger.info("Starting trainer.fit()...")
    trainer.fit(model, train_loader, val_loader)

    logger.info("Training complete. Checkpoints saved to: %s", checkpoint_dir)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.exception("Training process failed!")
        raise