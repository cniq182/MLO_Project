from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from en_es_translation.model import Model


def test_model_init_validates_hparams() -> None:
    # lr <= 0
    with (
        patch("en_es_translation.model.T5Tokenizer.from_pretrained"),
        patch("en_es_translation.model.T5ForConditionalGeneration.from_pretrained"),
    ):
        try:
            Model(lr=0.0)
            assert False, "Expected ValueError for lr <= 0"
        except ValueError:
            pass

        # batch_size <= 0
        try:
            Model(batch_size=0)
            assert False, "Expected ValueError for batch_size <= 0"
        except ValueError:
            pass


@patch("en_es_translation.model.T5ForConditionalGeneration.from_pretrained")
@patch("en_es_translation.model.T5Tokenizer.from_pretrained")
def test_forward_returns_list_of_strings(
    mock_tok_from_pretrained, mock_t5_from_pretrained
) -> None:
    # Fake tokenizer output: has .to(self.device), .input_ids, .attention_mask
    fake_inputs = MagicMock()
    fake_inputs.input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    fake_inputs.attention_mask = torch.ones_like(fake_inputs.input_ids)

    def _to(_device):
        return fake_inputs

    fake_inputs.to.side_effect = _to

    tokenizer = MagicMock()
    tokenizer.return_value = fake_inputs
    tokenizer.decode.side_effect = ["hola", "buenas"]
    mock_tok_from_pretrained.return_value = tokenizer

    # Fake model generate output: two sequences
    t5 = MagicMock()
    t5.generate.return_value = [torch.tensor([10, 11]), torch.tensor([12, 13])]
    mock_t5_from_pretrained.return_value = t5

    model = Model()
    model.eval()

    out = model(["Hello", "Good morning"], max_new_tokens=8)

    assert isinstance(out, list)
    assert out == ["hola", "buenas"]

    # Ensure prefix is applied and tokenizer called
    called_texts = tokenizer.call_args[0][0]
    assert all(s.startswith(model.prefix) for s in called_texts)


@patch("en_es_translation.model.T5ForConditionalGeneration.from_pretrained")
@patch("en_es_translation.model.T5Tokenizer.from_pretrained")
def test_training_step_logs_and_returns_loss(
    mock_tok_from_pretrained, mock_t5_from_pretrained
) -> None:
    # Tokenizer isn't used in training_step, but init needs it
    mock_tok_from_pretrained.return_value = MagicMock()

    # Fake forward outputs with .loss
    loss = torch.tensor(1.234, dtype=torch.float32)
    t5 = MagicMock()
    t5.return_value = SimpleNamespace(loss=loss)
    mock_t5_from_pretrained.return_value = t5

    model = Model(batch_size=2)

    batch = {
        "input_ids": torch.ones((2, 4), dtype=torch.long),
        "attention_mask": torch.ones((2, 4), dtype=torch.long),
        "labels": torch.ones((2, 4), dtype=torch.long),
    }

    out_loss = model.training_step(batch, batch_idx=0)
    assert torch.is_tensor(out_loss)
    assert float(out_loss.item()) == float(loss.item())


@patch("en_es_translation.model.T5ForConditionalGeneration.from_pretrained")
@patch("en_es_translation.model.T5Tokenizer.from_pretrained")
def test_configure_optimizers_returns_adamw(
    mock_tok_from_pretrained,
    mock_t5_from_pretrained,
) -> None:
    mock_tok_from_pretrained.return_value = MagicMock()
    mock_t5_from_pretrained.return_value = MagicMock()

    model = Model(lr=1e-4)

    # Add at least one real parameter so the optimizer has something to optimize
    model.dummy_param = torch.nn.Parameter(torch.tensor(1.0))

    opt = model.configure_optimizers()

    assert isinstance(opt, torch.optim.AdamW)
    assert opt.defaults["lr"] == 1e-4
