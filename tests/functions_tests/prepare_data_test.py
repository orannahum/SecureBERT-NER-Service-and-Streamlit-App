
# Test cases\
from ner_transformers_and_utilities.functions import prepare_data
import pytest

def test_normal_case():
    raw_data = "Hello B-PER\nworld O\n\nI O\nam O\nJohn B-PER\n"
    expected = [
        {"tokens": ["Hello", "world"], "labels": ["B-PER", "O"]},
        {"tokens": ["I", "am", "John"], "labels": ["O", "O", "B-PER"]},
    ]
    result = prepare_data(raw_data)
    assert result == expected


def test_empty_input():
    raw_data = ""
    expected = []
    result = prepare_data(raw_data)
    assert result == expected

def test_no_tokens_labels():
    raw_data = "The O\nadmin@338 B-HackOrg\nhas O"
    print(prepare_data(raw_data))
    result = prepare_data(raw_data)
    expected = [{'tokens': ['The', 'admin@338', 'has'], 'labels': ['O', 'B-HackOrg', 'O']}]
    assert result == expected

def test_multiple_sentences():
    raw_data = "Hello B-PER\nworld O\n\nI O\nam O\n\nJohn B-PER\nDoe I-PER\n"
    expected = [
        {"tokens": ["Hello", "world"], "labels": ["B-PER", "O"]},
        {"tokens": ["I", "am"], "labels": ["O", "O"]},
        {"tokens": ["John", "Doe"], "labels": ["B-PER", "I-PER"]},
    ]
    result = prepare_data(raw_data)
    assert result == expected

if __name__ == "__main__":
    pytest.main()