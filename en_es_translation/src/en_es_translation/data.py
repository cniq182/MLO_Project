import zipfile
from pathlib import Path

import requests
import torch
import typer
from torch.utils.data import Dataset
from transformers import AutoTokenizer

URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/en-es.txt.zip"
MODEL_CHECKPOINT = "google-t5/t5-small"


class MyDataset(Dataset):
    """Dataset for fine-tuning T5 translation models."""

    def __init__(self, data_path: Path) -> None:
        self.data = torch.load(data_path, weights_only=False)

    def __len__(self) -> int:
        return len(self.data["input_ids"])

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.data["input_ids"][idx],
            "attention_mask": self.data["attention_mask"][idx],
            "labels": self.data["labels"][idx],
        }


def download_and_extract(raw_dir: Path) -> None:
    """Download the dataset if it doesn't exist and extract it."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    en_file = raw_dir / "OpenSubtitles.en-es.en"
    es_file = raw_dir / "OpenSubtitles.en-es.es"
    
    if en_file.exists() and es_file.exists():
        print("Raw data files already exist. Skipping download and extraction.")
        return

    zip_path = raw_dir / "en-es.txt.zip"
    if not zip_path.exists():
        print(f"Downloading data from {URL}...")
        response = requests.get(URL, stream=True)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

    print("Extracting data...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(raw_dir)
    print("Extraction complete.")


def preprocess(
    raw_dir: str = "en_es_translation/data/raw",
    processed_dir: str = "en_es_translation/data/processed",
    num_samples: int = 10000,
) -> None:
    """Download, extract, and tokenize data for T5 fine-tuning."""
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)

    download_and_extract(raw_path)
    processed_path.mkdir(parents=True, exist_ok=True)

    print(f"Initializing tokenizer: {MODEL_CHECKPOINT}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    print(f"Reading first {num_samples} samples...")
    en_file = raw_path / "OpenSubtitles.en-es.en"
    es_file = raw_path / "OpenSubtitles.en-es.es"

    inputs, targets = [], []
    prefix = "translate English to Spanish: "

    with open(en_file, encoding="utf-8") as f_en, open(es_file, encoding="utf-8") as f_es:
        for _, (en, es) in zip(range(num_samples), zip(f_en, f_es)):
            en, es = en.strip(), es.strip()
            if en and es:
                inputs.append(prefix + en)
                targets.append(es)

    print("Tokenizing data...")
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Tokenize targets (labels)
    labels = tokenizer(
        text_target=targets,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Replace padding token id with -100 to ignore it in loss
    tokenized_data = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": labels["input_ids"],
    }
    tokenized_data["labels"][tokenized_data["labels"] == tokenizer.pad_token_id] = -100

    output_file = processed_path / "train_data.pt"
    torch.save(tokenized_data, output_file)
    print(f"Preprocessed data saved to {output_file}")


if __name__ == "__main__":
    typer.run(preprocess)