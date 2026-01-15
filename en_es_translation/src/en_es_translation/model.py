from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.models import _MODEL_PATH


class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        batch_size: int = 8,
        max_source_length: int = 128,
        max_target_length: int = 128,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if not isinstance(lr, (float, int)):
            raise TypeError("Learning rate must be a float or int.")
        if lr <= 0:
            raise ValueError("Learning rate must be > 0.")

        if not isinstance(batch_size, int):
            raise TypeError("Batch size must be an int.")
        if batch_size <= 0:
            raise ValueError("Batch size must be > 0.")

        self.save_hyperparameters()

        self.tokenizer = T5Tokenizer.from_pretrained(
            "t5-small", cache_dir=_MODEL_PATH, model_max_length=512
        )
        self.t5 = T5ForConditionalGeneration.from_pretrained(
            "t5-small", cache_dir=_MODEL_PATH
        )

        self.lr = float(lr)
        self.batch_size = batch_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        # T5 works better with a task prefix
        self.prefix = "translate English to Spanish: "

    def forward(self, x: List[str], max_new_tokens: int = 64) -> List[str]:

        # Inference: input list of English strings -> output list of Spanish strings
        x = [self.prefix + s for s in x]

        input_ids = self.tokenizer(
            x,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
        ).input_ids.to(self.device)

        outputs = self.t5.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=4,
        )

        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    def _inference_training(self, batch: Dict, batch_idx: Optional[int] = None) -> torch.Tensor:
        # Expecting: batch["translation"]["en"], batch["translation"]["es"]. Translation from en to spanish
        src_texts = batch["translation"]["en"]
        tgt_texts = batch["translation"]["es"]

        src_texts = [self.prefix + s for s in src_texts]

        enc = self.tokenizer(
            src_texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_source_length,
        ).to(self.device)

        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_target_length,
            ).input_ids.to(self.device)

        # Ignore padding in loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        outputs = self.t5(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=labels,
        )
        return outputs.loss

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        loss = self._inference_training(batch, batch_idx)
        self.log("train_loss", loss, batch_size=self.batch_size, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        loss = self._inference_training(batch, batch_idx)
        self.log("val_loss", loss, batch_size=self.batch_size, prog_bar=True)
        return loss

    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        loss = self._inference_training(batch, batch_idx)
        self.log("test_loss", loss, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
