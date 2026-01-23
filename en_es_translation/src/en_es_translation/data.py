import zipfile
import logging
import os
import tempfile
from pathlib import Path

import requests
import torch
import typer
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# --- M14: Advanced Logging Setup ---
# 1. Define and create the log directory
log_dir = Path("logs_logging")
log_dir.mkdir(exist_ok=True)

# 2. Configure logging to use the subfolder
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        # This now points to logs_logging/data_processing.log
        logging.FileHandler(log_dir / "data_processing.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# GCS support
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    logger.warning("google-cloud-storage not available. GCS paths will not work.")

URL = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/moses/en-es.txt.zip"
MODEL_CHECKPOINT = "google-t5/t5-small"


class MyDataset(Dataset):
    def __init__(self, data_path: Path) -> None:
        logger.debug(f"Attempting to load tensor data from: {data_path}")
        self.data = torch.load(data_path, weights_only=False)
        logger.info(
            f"Loaded dataset from {data_path.name} with {len(self.data['input_ids'])} samples."
        )

    def __len__(self) -> int:
        return len(self.data["input_ids"])

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.data["input_ids"][idx],
            "attention_mask": self.data["attention_mask"][idx],
            "labels": self.data["labels"][idx],
        }


def _is_gcs_path(path: str) -> bool:
    """Check if path is a GCS path."""
    return path.startswith("gs://")


def _download_from_gcs(gcs_path: str, local_path: Path) -> Path:
    """Download a file from GCS to local path."""
    if not GCS_AVAILABLE:
        raise ImportError("google-cloud-storage is required for GCS paths. Install it with: pip install google-cloud-storage")
    
    # Parse GCS path: gs://bucket/path/to/file
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {gcs_path}")
    
    parts = gcs_path[5:].split("/", 1)  # Remove 'gs://' prefix
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""
    
    logger.info(f"Downloading from GCS: {gcs_path} -> {local_path}")
    
    # Create local directory if needed
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download from GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(str(local_path))
    
    logger.info(f"Successfully downloaded {blob_name} from GCS")
    return local_path


def _get_local_path(gcs_path: str, local_cache_dir: Path = None) -> Path:
    """Get local path for a file, downloading from GCS if necessary."""
    if not _is_gcs_path(gcs_path):
        return Path(gcs_path)
    
    # Use temp directory or provided cache directory
    if local_cache_dir is None:
        local_cache_dir = Path(tempfile.gettempdir()) / "gcs_cache"
    
    local_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create local path preserving structure
    # gs://bucket/data/processed/train_data.pt -> /tmp/gcs_cache/bucket/data/processed/train_data.pt
    parts = gcs_path[5:].split("/")  # Remove 'gs://' prefix
    bucket_name = parts[0]
    file_path = "/".join(parts[1:])
    
    local_path = local_cache_dir / bucket_name / file_path
    
    # Download if not exists locally
    if not local_path.exists():
        _download_from_gcs(gcs_path, local_path)
    
    return local_path


def get_datasets(
    processed_dir: str = "en_es_translation/data/processed",
) -> tuple[MyDataset, MyDataset, MyDataset]:
    # Handle GCS paths
    if _is_gcs_path(processed_dir):
        logger.info(f"Detected GCS path: {processed_dir}")
        # For GCS, download files to local cache
        local_cache = Path(tempfile.gettempdir()) / "gcs_cache" / "datasets"
        train_file = _get_local_path(f"{processed_dir}/train_data.pt", local_cache)
        eval_file = _get_local_path(f"{processed_dir}/eval_data.pt", local_cache)
        test_file = _get_local_path(f"{processed_dir}/test_data.pt", local_cache)
    else:
        processed_path = Path(processed_dir)
        train_file = processed_path / "train_data.pt"
        eval_file = processed_path / "eval_data.pt"
        test_file = processed_path / "test_data.pt"

    logger.debug(f"Checking for processed files: train={train_file}, eval={eval_file}, test={test_file}")

    # Check if files exist (for local paths, trigger preprocessing if missing)
    if not _is_gcs_path(processed_dir):
        if not (train_file.exists() and eval_file.exists() and test_file.exists()):
            logger.warning("Processed files missing! Initiating preprocessing...")
            preprocess(processed_dir=str(processed_path))
        else:
            logger.info("All processed files found on disk.")
    else:
        # For GCS paths, files should already be downloaded
        if not (train_file.exists() and eval_file.exists() and test_file.exists()):
            raise FileNotFoundError(
                f"Processed data files not found in GCS: {processed_dir}\n"
                "Please ensure train_data.pt, eval_data.pt, and test_data.pt exist in the GCS bucket."
            )

    return (
        MyDataset(train_file),
        MyDataset(eval_file),
        MyDataset(test_file),
    )


def download_and_extract(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    en_file = raw_dir / "OpenSubtitles.en-es.en"
    es_file = raw_dir / "OpenSubtitles.en-es.es"

    if en_file.exists() and es_file.exists():
        logger.debug("Raw English/Spanish files already exist. Skipping download.")
        return

    zip_path = raw_dir / "en-es.txt.zip"
    if not zip_path.exists():
        logger.info(f"Targeting download URL: {URL}")
        try:
            response = requests.get(URL, stream=True)
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Download successful.")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    logger.info("Extracting raw data...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(raw_dir)
    logger.debug("Files extracted successfully.")


def preprocess(
    raw_dir: str = "en_es_translation/data/raw",
    processed_dir: str = "en_es_translation/data/processed",
    num_samples: int = 50000,
) -> None:
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)

    download_and_extract(raw_path)
    processed_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading HuggingFace tokenizer: {MODEL_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    en_file = raw_path / "OpenSubtitles.en-es.en"
    es_file = raw_path / "OpenSubtitles.en-es.es"

    inputs, targets = [], []
    prefix = "translate English to Spanish: "

    logger.debug(f"Starting file stream for {num_samples} samples.")
    with (
        open(en_file, encoding="utf-8") as f_en,
        open(es_file, encoding="utf-8") as f_es,
    ):
        for i, (en, es) in enumerate(zip(f_en, f_es)):
            if i >= num_samples:
                break
            en, es = en.strip(), es.strip()
            if en and es:
                inputs.append(prefix + en)
                targets.append(es)
            if i % 10000 == 0 and i > 0:
                logger.debug(f"Processed {i} raw lines...")

    logger.info(f"Tokenizing {len(inputs)} sentence pairs...")
    model_inputs = tokenizer(
        inputs,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    labels = tokenizer(
        text_target=targets,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    tokenized_data = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": labels["input_ids"],
    }
    tokenized_data["labels"][tokenized_data["labels"] == tokenizer.pad_token_id] = -100

    num_total = len(tokenized_data["input_ids"])
    train_end = int(num_total * 0.8)
    eval_end = int(num_total * 0.9)

    splits = {
        "train_data.pt": {k: v[:train_end] for k, v in tokenized_data.items()},
        "eval_data.pt": {k: v[train_end:eval_end] for k, v in tokenized_data.items()},
        "test_data.pt": {k: v[eval_end:] for k, v in tokenized_data.items()},
    }

    for filename, data in splits.items():
        torch.save(data, processed_path / filename)
        logger.info(f"Split {filename} saved with {len(data['input_ids'])} items.")

    logger.info("Data preprocessing finished successfully.")


if __name__ == "__main__":
    typer.run(preprocess)
