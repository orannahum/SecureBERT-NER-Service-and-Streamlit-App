import pytest
from typing import List, Dict
from ner_transformers_and_utilities.functions import reverse_tokenization



# Test Cases
def test_reverse_tokenization():
    # Sample input
    predictions = [
        [0, 1, 2, -1],  # Prediction for the first example
        [0, 0, 0, 1, -1]  # Prediction for the second example
    ]

    # Mapping of token IDs back to labels
    reverse_map_tokens = {
        0: "O",
        1: "B-PER",
        2: "I-PER"
    }

    # Expected output
    expected_output = [
        ["O", "B-PER", "I-PER"],  # Corresponding labels for the first prediction
        ["O", "O", "O", "B-PER"]  # Corresponding labels for the second prediction
    ]

    # Call the function
    reversed_output = reverse_tokenization(predictions, reverse_map_tokens)

    # Verify the output
    assert reversed_output == expected_output

if __name__ == '__main__':
    pytest.main()
