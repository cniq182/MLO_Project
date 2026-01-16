from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sacrebleu import corpus_bleu

from model import Model
from data import MyDataset
from transformers import T5Tokenizer


def evaluate():
    # ---------------- paths ----------------
    checkpoint_dir = Path("en_es_translation/models/checkpoints")
    processed_dir = Path("en_es_translation/data/processed")

    # automatically pick latest checkpoint
    checkpoint_path = max(checkpoint_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    print(f"Loading checkpoint: {checkpoint_path}")

    # ---------------- load model ----------------
    model = Model.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # ---------------- load test data ----------------
    test_dataset = MyDataset(processed_dir / "test_data.pt")
    test_loader = DataLoader(test_dataset, batch_size=16)

    predictions = []
    references = []

    print(f"Number of test samples: {len(test_dataset)}")
    print("Running evaluation on test set...")

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model.t5.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                num_beams=4,
            )

            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # decode references (replace -100 with pad token id)
            labels = labels.clone()
            labels[labels == -100] = tokenizer.pad_token_id
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(preds)
            references.extend(refs)

    # ---------------- BLEU ----------------
    bleu = corpus_bleu(predictions, [references])
    print(f"\nBLEU score on test set: {bleu.score:.2f}")

    # ---------------- qualitative examples ----------------
    print("\nSample translations:")
    for i in range(5):
        print(f"\nExample {i+1}")
        print(f"PRED: {predictions[i]}")
        print(f"REF : {references[i]}")


if __name__ == "__main__":
    evaluate()
