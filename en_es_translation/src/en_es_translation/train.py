import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.profiler import profile, ProfilerActivity

from model import Model
from data import get_datasets


def train():
    processed_data_dir = "en_es_translation/data/processed"
    checkpoint_dir = "en_es_translation/models/checkpoints"

    batch_size = 16
    epochs = 3
    lr = 1e-4

    print("Loading datasets")
    train_set, eval_set, _ = get_datasets(processed_dir=processed_data_dir)

    # Use subsets to keep runtime reasonable
    train_set = Subset(train_set, range(min(len(train_set), 50000)))
    eval_set = Subset(eval_set, range(min(len(eval_set), 20000)))
    print(f"DEBUG: Running with {len(train_set)} training samples")

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

    print("Initializing model")
    model = Model(lr=lr, batch_size=batch_size)

    # --------------------------------------------------
    # 🔍 PROFILING BLOCK (M13)
    # Profile ONE training step (correct for text models)
    # --------------------------------------------------
    print("\nRunning PyTorch profiler on one training step")

    model.train()
    batch = next(iter(train_loader))

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
    ) as prof:
        model.training_step(batch, batch_idx=0)

    print(
        prof.key_averages()
        .table(sort_by="cpu_time_total", row_limit=10)
    )

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

    print("Starting training")
    trainer.fit(model, train_loader, val_loader)

    print("Training complete")


if __name__ == "__main__":
    train()
