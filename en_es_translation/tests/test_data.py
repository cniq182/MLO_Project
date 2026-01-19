from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch

from en_es_translation.data import MyDataset, get_datasets, preprocess


class _FakeTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        texts: List[str] | None = None,
        *,
        text_target: List[str] | None = None,
        max_length: int = 128,
        truncation: bool = True,
        padding: str = "max_length",
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        # Produce deterministic fake tensors with shape (N, max_length)
        if texts is not None:
            n = len(texts)
            input_ids = torch.ones((n, max_length), dtype=torch.long)
            attention_mask = torch.ones((n, max_length), dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        if text_target is not None:
            n = len(text_target)
            # labels include some pad tokens (0) so replacement to -100 can be tested
            labels = torch.ones((n, max_length), dtype=torch.long)
            labels[:, -1] = self.pad_token_id
            return {"input_ids": labels}

        raise ValueError("Either texts or text_target must be provided.")


def _write_raw_parallel_files(raw_dir: Path, n_lines: int = 10) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    en_file = raw_dir / "OpenSubtitles.en-es.en"
    es_file = raw_dir / "OpenSubtitles.en-es.es"

    en_lines = [f"Hello {i}\n" for i in range(n_lines)]
    es_lines = [f"Hola {i}\n" for i in range(n_lines)]
    en_file.write_text("".join(en_lines), encoding="utf-8")
    es_file.write_text("".join(es_lines), encoding="utf-8")


def test_mydataset_len_and_getitem(tmp_path: Path) -> None:
    data = {
        "input_ids": torch.randint(0, 10, (5, 8), dtype=torch.long),
        "attention_mask": torch.ones((5, 8), dtype=torch.long),
        "labels": torch.randint(0, 10, (5, 8), dtype=torch.long),
    }
    pt_path = tmp_path / "train_data.pt"
    torch.save(data, pt_path)

    ds = MyDataset(pt_path)
    assert len(ds) == 5

    item = ds[0]
    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}
    assert item["input_ids"].shape == (8,)
    assert item["attention_mask"].shape == (8,)
    assert item["labels"].shape == (8,)


def test_preprocess_creates_splits_without_downloading(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_raw_parallel_files(raw_dir, n_lines=10)

    # Avoid network: make download_and_extract a no-op
    monkeypatch.setattr("en_es_translation.data.download_and_extract", lambda _: None)

    # Avoid HF download: patch AutoTokenizer.from_pretrained to return fake tokenizer
    monkeypatch.setattr(
        "en_es_translation.data.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: _FakeTokenizer(),
    )

    preprocess(raw_dir=str(raw_dir), processed_dir=str(processed_dir), num_samples=10)

    train_path = processed_dir / "train_data.pt"
    eval_path = processed_dir / "eval_data.pt"
    test_path = processed_dir / "test_data.pt"

    assert train_path.exists()
    assert eval_path.exists()
    assert test_path.exists()

    train_obj = torch.load(train_path, map_location="cpu", weights_only=False)
    eval_obj = torch.load(eval_path, map_location="cpu", weights_only=False)
    test_obj = torch.load(test_path, map_location="cpu", weights_only=False)

    # Check keys
    for obj in (train_obj, eval_obj, test_obj):
        assert set(obj.keys()) == {"input_ids", "attention_mask", "labels"}
        assert obj["input_ids"].ndim == 2
        assert obj["attention_mask"].ndim == 2
        assert obj["labels"].ndim == 2

    # Check 80/10/10 split sizes for 10 samples -> 8/1/1
    assert train_obj["input_ids"].shape[0] == 8
    assert eval_obj["input_ids"].shape[0] == 1
    assert test_obj["input_ids"].shape[0] == 1

    # Check padding replacement in labels (last position should be -100)
    assert (train_obj["labels"][:, -1] == -100).all()


def test_get_datasets_calls_preprocess_if_missing(tmp_path: Path, monkeypatch) -> None:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    called: dict[str, Any] = {"count": 0}

    def _fake_preprocess(*_args, **kwargs) -> None:
        called["count"] += 1
        out_dir = Path(kwargs["processed_dir"])
        # write minimal pt files so MyDataset can load them
        for name, n in [("train_data.pt", 2), ("eval_data.pt", 1), ("test_data.pt", 1)]:
            obj = {
                "input_ids": torch.ones((n, 4), dtype=torch.long),
                "attention_mask": torch.ones((n, 4), dtype=torch.long),
                "labels": torch.ones((n, 4), dtype=torch.long),
            }
            torch.save(obj, out_dir / name)

    monkeypatch.setattr("en_es_translation.data.preprocess", _fake_preprocess)

    train_ds, eval_ds, test_ds = get_datasets(processed_dir=str(processed_dir))

    assert called["count"] == 1
    assert len(train_ds) == 2
    assert len(eval_ds) == 1
    assert len(test_ds) == 1
