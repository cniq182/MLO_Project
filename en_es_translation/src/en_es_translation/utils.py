from pathlib import Path
import torch
from .model import Model


def load_model(
    checkpoint_dir: str | Path = "en_es_translation/models/checkpoints",
) -> tuple[Model, torch.device]:
    checkpoint_dir = Path(checkpoint_dir)

    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoint files found in {checkpoint_dir}. "
            "Please train a model first."
        )

    checkpoint_path = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading checkpoint: {checkpoint_path}")

    model = Model.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on device: {device}")

    return model, device
