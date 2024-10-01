import torch
import pytest
from typing import List, Dict, Any
from transformers import AutoTokenizer
from ner_transformers_and_utilities.functions import tokenize_and_map_labels
import warnings

# Ignore warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*clean_up_tokenization_spaces.*")

# Test Cases
def test_tokenize_and_map_labels():
    # Sample tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Example input
    examples = {
        "tokens": [
            ["Hello", "world"],
            ["This", "is", "a", "test"]
        ],
        "labels": [
            ["O", "O"],  # Token-level labels for the first sentence
            ["O", "O", "O", "O"]  # Token-level labels for the second sentence
        ]
    }

    # Mapping of labels to token IDs
    map_tokens = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2
    }

    # Call the function
    tokenized_outputs = tokenize_and_map_labels(examples, tokenizer, map_tokens)

    # Verify the output
    assert "input_ids" in tokenized_outputs
    assert "attention_mask" in tokenized_outputs
    assert "labels" in tokenized_outputs

    # Check the shape of input_ids and labels
    assert tokenized_outputs["input_ids"].shape == (2, 128)  # 2 sentences, max length of 128
    assert tokenized_outputs["labels"].shape == (2, 128)  # 2 sentences, max length of 128

# Additional test cases can be added here

if __name__ == '__main__':
    pytest.main()
