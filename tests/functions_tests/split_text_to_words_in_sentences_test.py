import pytest
from typing import List, Dict
from ner_transformers_and_utilities.functions import split_text_to_words_in_sentences

# Test Cases
def test_split_text_to_words_in_sentences():
    text = 'The O\nadmin@338 B-HackOrg\nhas O\nlargely O\ntargeted O'
    expected_output = [
        {
            'tokens': ['The', 'O', 'admin@338', 'B-HackOrg', 'has', 'O', 'largely', 'O', 'targeted', 'O'],
            'labels': ['-100'] * 10  # 10 tokens, each with a label of '-100'
        }
    ]

    result = split_text_to_words_in_sentences(text)

    # Assertions
    assert len(result) == 1  # Expecting one sentence in the output
    assert result[0]['tokens'] == expected_output[0]['tokens']
    assert result[0]['labels'] == expected_output[0]['labels']

if __name__ == '__main__':
    pytest.main()
