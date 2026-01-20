from typing import Dict, List
import pytorch_lightning as pl
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        batch_size: int = 16,
        max_source_length: int = 128,
        max_target_length: int = 128,
        model_name: str = "google-t5/t5-small",
        prefix: str = "translate English to Spanish: ",
        max_new_tokens: int = 64,
        num_beams: int = 4,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Validation for safety
        if lr <= 0:
            raise ValueError("Learning rate must be > 0.")
        if batch_size <= 0:
            raise ValueError("Batch size must be > 0.")

        self.save_hyperparameters()

        # Model definition
        print(f"--- Verification: Loading {model_name} weights ---")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)

        self.lr = lr
        self.batch_size = batch_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_name = model_name
        self.prefix = prefix
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        print("--- Model successfully initialized ---")

    def forward(self, x: List[str], max_new_tokens: int = None) -> List[str]:
        """
        Inference pass: Takes a list of English strings and returns Spanish strings.
        Used for evaluation and real-world testing.

        Args:
            x: List of input English strings
            max_new_tokens: Maximum number of tokens to generate. If None, uses self.max_new_tokens
        """
        # Use instance default if not provided
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # Add prefix to each sentence
        prefixed_x = [self.prefix + s for s in x]

        inputs = self.tokenizer(
            prefixed_x,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
        ).to(self.device)

        outputs = self.t5.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=self.num_beams,
        )

        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Logic for a single training iteration.
        Expects keys: 'input_ids', 'attention_mask', 'labels'
        """
        outputs = self.t5(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        # Periodic console logging for verification
        if batch_idx % 10 == 0:
            print(f"Step {batch_idx} | Train Loss: {loss.item():.4f}")

        self.log(
            "train_loss", loss, batch_size=self.batch_size, prog_bar=True, on_step=True
        )
        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Logic for a single validation iteration."""
        outputs = self.t5(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log(
            "val_loss",
            outputs.loss,
            batch_size=self.batch_size,
            prog_bar=True,
            on_epoch=True,
        )
        return outputs.loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Logic for a single test iteration."""
        outputs = self.t5(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log("test_loss", outputs.loss, batch_size=self.batch_size)
        return outputs.loss

    def configure_optimizers(self):
        """Initializes the optimizer (AdamW)."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# --- Built-in Verification Block ---
if __name__ == "__main__":
    # Check if we can initialize and run a dummy pass
    print("Testing Model initialization...")
    test_model = Model(lr=1e-4, batch_size=2)

    # Check weights: print a tiny slice of the embedding layer
    weights_slice = test_model.t5.shared.weight[0][:5]
    print(f"Weights check (embedding slice): {weights_slice}")

    # Test dummy inference
    en_sentences = ["How are you?", "This is a machine translation test."]
    print(f"Testing inference with: {en_sentences}")
    results = test_model(en_sentences)
    for en, es in zip(en_sentences, results):
        print(f"  EN: {en} -> ES: {es}")

    print("\nModel script verification complete.")
