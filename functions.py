# Function to prepare the data into sentences
from transformers import EvalPrediction
import numpy as np
from sklearn.metrics import precision_score, recall_score
import pandas as pd
import torch
import re


def prepare_data(raw_data):
    # Split the raw data into sentences based on double newlines
    sentences = raw_data.strip().split("\n\n")
    
    prepared_data = []
    
    for sentence in sentences:
        tokens_labels = sentence.strip().split("\n")
        tokens = []
        labels = []
        
        for token_label in tokens_labels:
            token_label = token_label.strip()  # Strip whitespace
            
            # Skip empty lines
            if not token_label:
                continue
            
            # Ensure there are two parts to unpack
            if ' ' in token_label:
                token, label = token_label.rsplit(" ", 1)  # Split on the last space
                tokens.append(token)
                labels.append(label)

        # Add only non-empty sentences
        if tokens and labels:
            prepared_data.append({"tokens": tokens, "labels": labels})

    return prepared_data


def compute_metrics(p: EvalPrediction):
    predictions, labels = p.predictions, p.label_ids

    # Get the predicted labels
    preds = np.argmax(predictions, axis=2)

    # Flatten the arrays and remove padding values (-1)
    true_labels = []
    pred_labels = []

    for label, pred in zip(labels, preds):
        # Ignore the label -1
        true_labels.extend([l for l in label if l != -1])
        pred_labels.extend([p for p, l in zip(pred, label) if l != -1])

    # Calculate the precision  and recall
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    return {"precision": np.round(precision,4), "recall": np.round(recall,4) }



def plot_training_logs(train_logs):
    import matplotlib.pyplot as plt

    # Convert logs to DataFrame
    logs_df = pd.DataFrame(train_logs)

    # Ensure the DataFrame is not empty
    if not logs_df.empty:
        # Extract unique epochs
        epochs = logs_df['epoch'].unique()
        
        # Initialize lists to hold average loss values
        train_loss_per_epoch = []
        eval_loss_per_epoch = []
        eval_precision_per_epoch = []
        eval_recall_per_epoch = []
        
        # Loop through each epoch and calculate average losses
        for epoch in epochs:
            epoch_logs = logs_df[logs_df['epoch'] == epoch]
            train_loss_per_epoch.append(epoch_logs['loss'].mean() if 'loss' in epoch_logs.columns else None)
            
            eval_loss_per_epoch.append(epoch_logs['eval_loss'].mean() if 'eval_loss' in epoch_logs.columns else None)
            eval_precision_per_epoch.append(epoch_logs['eval_precision'].mean() if 'eval_precision' in epoch_logs.columns else None)
            eval_recall_per_epoch.append(epoch_logs['eval_recall'].mean() if 'eval_recall' in epoch_logs.columns else None)

        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'epoch': epochs,
            'train_loss': train_loss_per_epoch,
            'eval_loss': eval_loss_per_epoch,
            'eval_precision': eval_precision_per_epoch,
            'eval_recall': eval_recall_per_epoch
        })
        
        # Interpolate to fill NaN values
        plot_df = plot_df.interpolate(method='linear')

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(plot_df['epoch'], plot_df['train_loss'], label='Training Loss', color='blue', marker='o')
        plt.plot(plot_df['epoch'], plot_df['eval_loss'], label='Validation Loss', color='orange', marker='o', linestyle='dashed')
        plt.plot(plot_df['epoch'], plot_df['eval_precision'], label='Validation Precision', color='green', marker='o', linestyle='dashed')
        plt.plot(plot_df['epoch'], plot_df['eval_recall'], label='Validation Recall', color='red', marker='o', linestyle='dashed')

        plt.title('Training and Validation Metrics Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.xticks(epochs)  # Set x-ticks to be the epoch numbers
        plt.legend()
        plt.grid()
        plt.show()
    else:
        print("No logs found.")


def tokenize_and_map_labels(examples, tokenizer, map_tokens):
    tokenized_inputs = tokenizer(examples["tokens"], 
                                  truncation=True, 
                                  is_split_into_words=True, 
                                  padding="max_length", 
                                  max_length=128) # max tokens on all data in 65 so we can set max_length to 128
    
    # Initialize labels array
    tokenized_inputs['labels'] = []
    for label in examples["labels"]:
        tokenized_label = [map_tokens.get(l, -1) for l in label]
        padded_label = tokenized_label + [-1] * (128 - len(tokenized_label))  # Pad with -1
        padded_label = padded_label[:128]  # Truncate if too long
        tokenized_inputs['labels'].append(padded_label)

    tokenized_inputs = {key: torch.tensor(val) for key, val in tokenized_inputs.items()}
    
    return tokenized_inputs        


def reverse_tokenization(predictions, reverse_map_tokens):
    reversed_labels = []
    
    for pred in predictions:
        # Convert the predicted IDs to labels, filtering out the -1 padding
        labels = [reverse_map_tokens.get(label_id, 'O') for label_id in pred if label_id != -1]
        reversed_labels.append(labels)
    
    return reversed_labels 


def split_text_to_words_in_sentences(text):
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    # i want list of dicts [{'tokens':[...]}, {'tokens':[...]}]
    words_by_sentence = []
    for sentence in sentences:
        # Split the sentence into words
        words = sentence.split()
        # Create a dictionary with the tokens
        words_by_sentence.append({'tokens': words, 'labels': ['-100']*len(words)})
    
    return words_by_sentence    





def predict_pre_train_model(tokenized_input_example, model, pipeline_from_predictions_to_labels, device=torch.device("cpu")):
    model.to(device)

    # Convert input_ids and attention_mask to torch tensors and move to MPS
    input_ids_tensor = torch.tensor(tokenized_input_example['input_ids']).to(device)
    attention_mask_tensor = torch.tensor(tokenized_input_example['attention_mask']).to(device)

    with torch.no_grad():  # Disable gradient calculation
        outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
        logits = outputs.logits  # Get the logits

    predictions = logits.argmax(dim=2).cpu().numpy()  # Get the predicted class indices
    predictions = [pred[pred != -100] for pred in predictions]
    predict_labels = pipeline_from_predictions_to_labels.transform(predictions)

    return predict_labels



def calculate_metrics(tokenized_input_example, predict_labels, reversed_map_token):
    y_pred = []
    y_true = []
        # Example of how to print the results
    for i in range(len(tokenized_input_example)):
        number_of_tokens = len(tokenized_input_example[i]['tokens'])
        original_labels = tokenized_input_example[i]['labels'][:number_of_tokens]
        predicted_labels = predict_labels[i][:number_of_tokens]
        y_pred.append(predicted_labels)
        y_true.append(original_labels)
    y_pred = [label for labels in y_pred for label in labels]
    y_true = [label for labels in y_true for label in labels]
    #map all_true_labels_flat wiht reversed_map_token
    y_true = [reversed_map_token[label] for label in y_true]

    # Calculate precision and recall
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' or 'micro' as needed
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' or 'micro' as needed
    return precision, recall, y_true, y_pred