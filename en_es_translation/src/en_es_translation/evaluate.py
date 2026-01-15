import sys
from pathlib import Path
import torch

from model import Model

def test_translation():
    # 1. Path to your best checkpoint
    # Update the filename if it differs in your folder
    checkpoint_path = "en_es_translation/models/checkpoints/test-run-epoch=01.ckpt"
    
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    # 2. Load the model from the checkpoint
    print(f"--- Loading model from {checkpoint_path} ---")
    model = Model.load_from_checkpoint(checkpoint_path)
    model.eval() # Set to evaluation mode
    model.freeze() # Freeze weights for inference

    # 3. Test Sentences
    test_sentences = [
        "How are you?",
        "The cat is on the table.",
        "I love machine learning.",
        "This is a test of the translation system."
    ]

    print("\n--- Testing Translations ---")
    translations = model(test_sentences)

    for en, es in zip(test_sentences, translations):
        print(f"EN: {en}")
        print(f"ES: {es}")
        print("-" * 20)

if _name_ == "_main_":
    test_translation()