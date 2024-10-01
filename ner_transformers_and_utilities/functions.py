# Function to prepare the data into sentences
from transformers import EvalPrediction
import numpy as np
from sklearn.metrics import precision_score, recall_score
import pandas as pd
import torch
import re
from typing import Dict, List, Any, Tuple



def prepare_data(raw_data: str) -> List[Dict[str, List[str]]]:
    """
    Process raw text data to extract tokens and their corresponding labels.

    Args:
        raw_data (str): A raw string containing sentences. Sentences are separated
                         by double newlines, and each token-label pair is on a new line,
                         separated by a space.

    Returns:
        List[Dict[str, List[str]]]: A list of dictionaries where each dictionary
                                      contains:
                                      - "tokens": A list of tokens (words).
                                      - "labels": A list of corresponding labels for each token.
    """
    sentences = raw_data.strip().split("\n\n")
    
    prepared_data = []
    
    for sentence in sentences:
        tokens_labels = sentence.strip().split("\n")
        tokens = []
        labels = []
        
        for token_label in tokens_labels:
            token_label = token_label.strip()  
            
            if not token_label:
                continue
            
            if ' ' in token_label:
                token, label = token_label.rsplit(" ", 1) 
                tokens.append(token)
                labels.append(label)

        if tokens and labels:
            prepared_data.append({"tokens": tokens, "labels": labels})

    return prepared_data


def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    """
    Compute precision and recall for model predictions.

    Args:
        p (EvalPrediction): An instance of EvalPrediction containing predictions
                            and labels.

    Returns:
        Dict[str, float]: A dictionary with precision and recall values.
    """
    predictions, labels = p.predictions, p.label_ids
    preds = np.argmax(predictions, axis=2)
    true_labels = []
    pred_labels = []

    for label, pred in zip(labels, preds):
        true_labels.extend([l for l in label if l != -1])
        pred_labels.extend([p for p, l in zip(pred, label) if l != -1])

    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    return {"precision": np.round(precision, 4), "recall": np.round(recall, 4)}



def plot_training_logs(train_logs: List[Dict[str, Any]]) -> None:
    """
    Plot training and validation metrics over epochs.

    Args:
        train_logs (List[Dict[str, Any]]): A list of dictionaries containing training
                                             logs with metrics for each epoch.
    
    Returns:
        None: This function does not return a value; it displays the plot.
    """
    import matplotlib.pyplot as plt

    logs_df = pd.DataFrame(train_logs)
    
    if not logs_df.empty:
        epochs = logs_df['epoch'].unique()
        train_loss_per_epoch = []
        eval_loss_per_epoch = []
        eval_precision_per_epoch = []
        eval_recall_per_epoch = []

        for epoch in epochs:
            epoch_logs = logs_df[logs_df['epoch'] == epoch]
            train_loss_per_epoch.append(epoch_logs['loss'].mean() if 'loss' in epoch_logs.columns else None)
            eval_loss_per_epoch.append(epoch_logs['eval_loss'].mean() if 'eval_loss' in epoch_logs.columns else None)
            eval_precision_per_epoch.append(epoch_logs['eval_precision'].mean() if 'eval_precision' in epoch_logs.columns else None)
            eval_recall_per_epoch.append(epoch_logs['eval_recall'].mean() if 'eval_recall' in epoch_logs.columns else None)

        plot_df = pd.DataFrame({
            'epoch': epochs,
            'train_loss': train_loss_per_epoch,
            'eval_loss': eval_loss_per_epoch,
            'eval_precision': eval_precision_per_epoch,
            'eval_recall': eval_recall_per_epoch
        })
        
        plot_df = plot_df.interpolate(method='linear')

        plt.figure(figsize=(12, 6))
        plt.plot(plot_df['epoch'], plot_df['train_loss'], label='Training Loss', color='blue', marker='o')
        plt.plot(plot_df['epoch'], plot_df['eval_loss'], label='Validation Loss', color='orange', marker='o', linestyle='dashed')
        plt.plot(plot_df['epoch'], plot_df['eval_precision'], label='Validation Precision', color='green', marker='o', linestyle='dashed')
        plt.plot(plot_df['epoch'], plot_df['eval_recall'], label='Validation Recall', color='red', marker='o', linestyle='dashed')

        plt.title('Training and Validation Metrics Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.xticks(epochs)
        plt.legend()
        plt.grid()
        plt.show()
    else:
        print("No logs found.")


def tokenize_and_map_labels(examples: Dict[str, List[str]], tokenizer: Any, map_tokens: Dict[str, int]) -> Dict[str, torch.Tensor]:
    """
    Tokenize input sentences and map their labels to token IDs.

    Args:
        examples (Dict[str, List[str]]): A dictionary containing tokenized inputs
                                           with "tokens" and corresponding "labels".
        tokenizer (Any): The tokenizer used to convert tokens into input IDs.
        map_tokens (Dict[str, int]): A dictionary mapping label strings to token IDs.
    
    Returns:
        Dict[str, torch.Tensor]: A dictionary containing tokenized inputs and labels
                                  as PyTorch tensors.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )
    
    tokenized_inputs['labels'] = []
    for label in examples["labels"]:
        tokenized_label = [map_tokens.get(l, -1) for l in label]
        padded_label = tokenized_label + [-1] * (128 - len(tokenized_label))
        padded_label = padded_label[:128]
        tokenized_inputs['labels'].append(padded_label)

    tokenized_inputs = {key: torch.tensor(val) for key, val in tokenized_inputs.items()}
    
    return tokenized_inputs     


def reverse_tokenization(predictions: List[List[int]], reverse_map_tokens: Dict[int, str]) -> List[List[str]]:
    """
    Reverse the tokenization process by converting predicted IDs back to labels.

    Args:
        predictions (List[List[int]]): A list of lists containing predicted token IDs.
        reverse_map_tokens (Dict[int, str]): A dictionary mapping token IDs to their corresponding labels.

    Returns:
        List[List[str]]: A list of lists containing the corresponding labels for each prediction.
    """
    reversed_labels = []
    
    for pred in predictions:
        labels = [reverse_map_tokens.get(label_id, 'O') for label_id in pred if label_id != -1]
        reversed_labels.append(labels)
    
    return reversed_labels


def split_text_to_words_in_sentences(text: str) -> List[Dict[str, List[str]]]:
    """
    Split the input text into sentences and further split each sentence into words.

    Args:
        text (str): The input text to be split into sentences and words.

    Returns:
        List[Dict[str, List[str]]]: A list of dictionaries where each dictionary contains a 
                                     'tokens' key with a list of words from each sentence, 
                                     and a 'labels' key with a list of '-100' labels 
                                     for each word.
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    words_by_sentence = []
    
    for sentence in sentences:
        words = sentence.split()
        words_by_sentence.append({'tokens': words, 'labels': ['-100'] * len(words)})
    
    return words_by_sentence



def predict_pre_train_model(
    tokenized_input_example: dict, 
    model: torch.nn.Module, 
    pipeline_from_predictions_to_labels, 
    device: torch.device = torch.device("cpu")
) -> List:
    """
    Make predictions using a pre-trained model on tokenized input examples.

    Args:
        tokenized_input_example (dict): The input example containing 'input_ids' and 'attention_mask'.
        model (torch.nn.Module): The pre-trained model to be used for predictions.
        pipeline_from_predictions_to_labels: A pipeline to transform predictions to labels.
        device (torch.device, optional): The device to run the model on (default is CPU).

    Returns:
        List: A list of predicted labels for the input example.
    """
    model.to(device)
    
    input_ids_tensor = torch.tensor(tokenized_input_example['input_ids']).to(device)
    attention_mask_tensor = torch.tensor(tokenized_input_example['attention_mask']).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
        logits = outputs.logits

    predictions = logits.argmax(dim=2).cpu().numpy()
    predictions = [pred[pred != -100] for pred in predictions]
    predict_labels = pipeline_from_predictions_to_labels.transform(predictions)

    return predict_labels



def calculate_metrics(
    tokenized_input_example: List[Dict], 
    predict_labels: List[List[str]], 
    reversed_map_token: Dict[int, str]
) -> Tuple[float, float, List[str], List[str]]:
    """
    Calculate precision and recall metrics based on the predicted and true labels.

    Args:
        tokenized_input_example (List[Dict]): A list of tokenized input examples containing 'tokens' and 'labels'.
        predict_labels (List[List[str]]): A list of predicted labels for the input examples.
        reversed_map_token (Dict[int, str]): A mapping from label IDs to label names.

    Returns:
        Tuple[float, float, List[str], List[str]]: A tuple containing precision, recall, true labels, and predicted labels.
    """
    y_pred = []
    y_true = []

    for i in range(len(tokenized_input_example)):
        number_of_tokens = len(tokenized_input_example[i]['tokens'])
        original_labels = tokenized_input_example[i]['labels'][:number_of_tokens]
        predicted_labels = predict_labels[i][:number_of_tokens]
        y_pred.append(predicted_labels)
        y_true.append(original_labels)

    y_pred = [label for labels in y_pred for label in labels]
    y_true = [label for labels in y_true for label in labels]
    y_true = [reversed_map_token[label] for label in y_true]

    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    return precision, recall, y_true, y_pred