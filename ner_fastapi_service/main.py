import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from ner_transformers_and_utilities.data_processing_transformers import (
    DatasetFromListTransformer,
    MapAndTokenizeTransformer,
    ReverseTokenizationTransformer,
)
from ner_transformers_and_utilities.functions import split_text_to_words_in_sentences, predict_pre_train_model
from transformers import AutoModelForTokenClassification
import os
import json

# Define FastAPI app
app = FastAPI()


model_ner_index = None
model_ner_dir = None
model_ner = None

model_cyner_index = None
model_cyner_dir = None
model_cyner = None

dnrti_to_category = None


def load_model(models_name):
    global model_cyner_index, model_cyner_dir, model_cyner
    if model_cyner_index is None:
        model_cyner_index = max(
            (int(file.split("_")[1]) for file in os.listdir(f"models/{models_name}") if file.startswith("model_")),
            default=0
        )
        model_cyner_dir = f"models/{models_name}/model_{model_cyner_index}"
        model_cyner = AutoModelForTokenClassification.from_pretrained(model_cyner_dir)
        model_cyner.eval()

    # Load pipelines
    with open('pipelines/pipeline_from_predictions_to_labels.pkl', 'rb') as file:
        pipeline_from_predictions_to_labels = pickle.load(file)

    with open('pipelines/pipeline_from_input_to_lentokenized_data.pkl', 'rb') as file:
        pipeline_from_input_to_lentokenized_data = pickle.load(file)

    return pipeline_from_input_to_lentokenized_data, pipeline_from_predictions_to_labels

def load_dnrti_to_category():
    global dnrti_to_category
    if dnrti_to_category is None:
        with open('dnrti_to_category.json', 'r') as file:
            dnrti_to_category = json.load(file)
    return dnrti_to_category

# Define the input model
class TextInput(BaseModel):
    text: str


# NER endpoint
@app.post("/ner/")
async def extract_entities_ner(input: TextInput):
    try:
        pipeline_from_input_to_lentokenized_data, pipeline_from_predictions_to_labels = load_model("securebert_ner")
        dnrti_to_category = load_dnrti_to_category()
        tokenized_input_example = pipeline_from_input_to_lentokenized_data.transform(input.text)
        predict_labels = predict_pre_train_model(tokenized_input_example, model_cyner, pipeline_from_predictions_to_labels)
        predict_labels_no_padding = []
        predict_labels_no_padding_cat = []
        for i in range(len(tokenized_input_example)):
            tokens = tokenized_input_example[i]['tokens']
            len_of_tokens = len(tokens)
            current_labels = predict_labels[i][:len_of_tokens]  # Get labels up to the length of tokens
            predict_labels_no_padding.append(current_labels)  # Append to the list
            current_labels_cat = [f"{x.split('-')[0]}-{dnrti_to_category.get(x.split('-')[1], 'O')}" if x != 'O' else 'O' for x in current_labels]
            predict_labels_no_padding_cat.append(current_labels_cat)
        return {"entities_dnrti": predict_labels_no_padding, "entities_cat": predict_labels_no_padding_cat}  # Return the properly structured list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# CyNER endpoint
@app.post("/cyner/")
async def extract_entities_cyner(input: TextInput):
    try:
        pipeline_from_input_to_lentokenized_data, pipeline_from_predictions_to_labels = load_model("securebert_cyner")
        dnrti_to_category = load_dnrti_to_category()
        tokenized_input_example = pipeline_from_input_to_lentokenized_data.transform(input.text)
        predict_labels = predict_pre_train_model(tokenized_input_example, model_cyner, pipeline_from_predictions_to_labels)
        predict_labels_no_padding = []
        predict_labels_no_padding_cat = []
        for i in range(len(tokenized_input_example)):
            tokens = tokenized_input_example[i]['tokens']
            len_of_tokens = len(tokens)
            current_labels = predict_labels[i][:len_of_tokens]  # Get labels up to the length of tokens
            predict_labels_no_padding.append(current_labels)  # Append to the list
            current_labels_cat = [f"{x.split('-')[0]}-{dnrti_to_category.get(x.split('-')[1], 'O')}" if x != 'O' else 'O' for x in current_labels]
            predict_labels_no_padding_cat.append(current_labels_cat)
        return {"entities_dnrti": predict_labels_no_padding, "entities_cat": predict_labels_no_padding_cat}  # Return the properly structured list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
