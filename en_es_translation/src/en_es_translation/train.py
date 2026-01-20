import os
import sys
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset # Added Subset
from pytorch_lightning.callbacks import ModelCheckpoint

from model import Model
from data import get_datasets

def train():
    processed_data_dir = "en_es_translation/data/processed"
    checkpoint_dir = "en_es_translation/models/checkpoints"
    
    batch_size = 16 # Lowered batch size for smaller test
    epochs = 3     # Just 2 epochs to see the transition
    lr = 1e-4

    print("Loading Datasets")
    train_set, eval_set, _ = get_datasets(processed_dir=processed_data_dir)

    
    # We take only the first 100 samples for training and 50 for validation
    train_set = Subset(train_set, range(min(len(train_set), 50000)))
    eval_set = Subset(eval_set, range(min(len(eval_set), 20000)))
    print(f"DEBUG: Running with {len(train_set)} training samples.")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(eval_set, batch_size=batch_size, num_workers=0)

    print("Initializing Model")
    model = Model(lr=lr, batch_size=batch_size)

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
        log_every_n_steps=1, # Log more often for small data
    )

    print("Starting Fast Test Training")
    trainer.fit(model, train_loader, val_loader)
    
    print(f"Test training complete.")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best model score: {checkpoint_callback.best_model_score}")

if __name__ == "__main__":
    train()