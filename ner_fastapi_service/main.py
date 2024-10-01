import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
from data_processing_transformers import (
    DatasetFromListTransformer,
    MapAndTokenizeTransformer,
    ReverseTokenizationTransformer,
)
from functions import split_text_to_words_in_sentences, predict_pre_train_model
from transformers import AutoModelForTokenClassification
import os

# Define FastAPI app
app = FastAPI()


model_ner_index = None
model_ner_dir = None
model_ner = None

model_cyner_index = None
model_cyner_dir = None
model_cyner = None

def load_model_ner():
    global model_ner_index, model_ner_dir, model_ner
    if model_ner_index is None:
        model_ner_index = max(
            (int(file.split("_")[1]) for file in os.listdir("models/securebert_ner") if file.startswith("model_")),
            default=0
        )
        model_ner_dir = f"models/securebert_ner/model_{model_ner_index}"
        model_ner = AutoModelForTokenClassification.from_pretrained(model_ner_dir)
        model_ner.eval()

    # Load pipelines
    with open('pipelines/pipeline_from_predictions_to_labels.pkl', 'rb') as file:
        pipeline_from_predictions_to_labels = pickle.load(file)

    with open('pipelines/pipeline_from_input_to_lentokenized_data.pkl', 'rb') as file:
        pipeline_from_input_to_lentokenized_data = pickle.load(file)

    return pipeline_from_input_to_lentokenized_data, pipeline_from_predictions_to_labels

def load_model_cyner():
    global model_cyner_index, model_cyner_dir, model_cyner
    if model_cyner_index is None:
        model_cyner_index = max(
            (int(file.split("_")[1]) for file in os.listdir("models/securebert_cyner") if file.startswith("model_")),
            default=0
        )
        model_cyner_dir = f"models/securebert_cyner/model_{model_cyner_index}"
        model_cyner = AutoModelForTokenClassification.from_pretrained(model_cyner_dir)
        model_cyner.eval()

    # Load pipelines
    with open('pipelines/pipeline_from_predictions_to_labels.pkl', 'rb') as file:
        pipeline_from_predictions_to_labels = pickle.load(file)

    with open('pipelines/pipeline_from_input_to_lentokenized_data.pkl', 'rb') as file:
        pipeline_from_input_to_lentokenized_data = pickle.load(file)

    return pipeline_from_input_to_lentokenized_data, pipeline_from_predictions_to_labels

# Define the input model
class TextInput(BaseModel):
    text: str


# NER endpoint
@app.post("/ner/")
async def extract_entities_ner(input: TextInput):
    try:
        pipeline_from_input_to_lentokenized_data, pipeline_from_predictions_to_labels = load_model_ner()
        
        tokenized_input_example = pipeline_from_input_to_lentokenized_data.transform(input.text)
        predict_labels = predict_pre_train_model(tokenized_input_example, model_ner, pipeline_from_predictions_to_labels)
        predict_labels_no_padding = []
        for i in range(len(tokenized_input_example)):
            tokens = tokenized_input_example[i]['tokens']
            len_of_tokens = len(tokens)
            current_labels = predict_labels[i][:len_of_tokens]  # Get labels up to the length of tokens
            predict_labels_no_padding.append(current_labels)  # Append to the list

        return {"entities_ner": predict_labels_no_padding}  # Return the properly structured list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# CyNER endpoint
@app.post("/cyner/")
async def extract_entities_cyner(input: TextInput):
    try:
        pipeline_from_input_to_lentokenized_data, pipeline_from_predictions_to_labels = load_model_cyner()
        
        tokenized_input_example = pipeline_from_input_to_lentokenized_data.transform(input.text)
        predict_labels = predict_pre_train_model(tokenized_input_example, model_cyner, pipeline_from_predictions_to_labels)
        predict_labels_no_padding = []
        for i in range(len(tokenized_input_example)):
            tokens = tokenized_input_example[i]['tokens']
            len_of_tokens = len(tokens)
            current_labels = predict_labels[i][:len_of_tokens]  # Get labels up to the length of tokens
            predict_labels_no_padding.append(current_labels)  # Append to the list

        return {"entities_cyner": predict_labels_no_padding}  # Return the properly structured list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
