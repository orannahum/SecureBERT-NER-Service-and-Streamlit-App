import pytest
from typing import List, Dict, Tuple
from sklearn.metrics import precision_score, recall_score

# Assuming calculate_metrics is defined in a module named 'metrics'
from ner_transformers_and_utilities.functions import calculate_metrics

def test_calculate_metrics():
    # Sample input data
    tokenized_input_example = [
        {'tokens': ['token1', 'token1', 'token1'], 'labels': [1, 1, 1]},
        {'tokens': ['token1', 'token1'], 'labels': [1, 1]}
    ]
    
    predict_labels = [
        ['token1', 'token1', 'token1'],  # Predicted labels as strings
        ['token1', 'token1']        # Predicted labels as strings
    ]
    
    reversed_map_token = {
        1: 'token1',
        2: 'token1',
        3: 'token1'
    }
    
    # Expected output calculations
    y_true = ['token1', 'token1', 'token1', 'token1', 'token1']
    y_pred = ['token1', 'token1', 'token1', 'token1', 'token1']
    
    # Calculate precision and recall using sklearn for expected values
    precision_expected = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_expected = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    # Call the function
    precision, recall, true_labels, predicted_labels = calculate_metrics(tokenized_input_example, predict_labels, reversed_map_token)

    # Assert the expected and actual values
    assert precision == precision_expected, f"Expected precision: {precision_expected}, got: {precision}"
    assert recall == recall_expected, f"Expected recall: {recall_expected}, got: {recall}"
    assert true_labels == y_true, f"Expected true labels: {y_true}, got: {true_labels}"
    assert predicted_labels == y_pred, f"Expected predicted labels: {y_pred}, got: {predicted_labels}"

# To run the tests, use the command:
# pytest -v path_to_this_test_file.py
