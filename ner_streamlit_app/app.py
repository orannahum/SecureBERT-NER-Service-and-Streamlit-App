import streamlit as st
import requests
import pandas as pd
import numpy as np
import pickle
import json
import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
from data_processing_transformers import (
    DatasetFromListTransformer,
    MapAndTokenizeTransformer,
    ReverseTokenizationTransformer,
)
from sklearn.metrics import precision_score, recall_score
from functions import predict_pre_train_model, calculate_metrics, split_text_to_words_in_sentences
import json
import time
import sklearn
import datasets

# Define the FastAPI endpoint
if __name__ == "__main__":
    # load json config


    print("version of skicit-learn")
    print(sklearn.__version__)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    with open('map_tokens.json', 'r') as file:
        map_token = json.load(file)

    with open('dnrti_to_category.json', 'r') as file:    
        dnrti_to_category = json.load(file)

    with open('config.json') as file:
        config = json.load(file)
    if 'model_ner_index' in config:    
         model_ner_index = config['model_ner_index']
    else:
        # take  the max in the model_ners
        model_ner_index = max(
            (int(file.split("_")[1]) for file in os.listdir("models/securebert_ner") if file.startswith("model_")),
            default=0
        )     
    if 'model_cyner_index' in config:
        model_cyner_index = config['model_cyner_index']
    else:
        # take  the max in the model_ners
        model_cyner_index = max(
            (int(file.split("_")[1]) for file in os.listdir("models/securebert_cyner") if file.startswith("model_")),
            default=0
        )    
    dnrti_to_category_df = pd.DataFrame(dnrti_to_category.items(), columns=['dnrti', 'category'])

    # Streamlit UI
    st.title("Named Entity Recognition (NER) Web Interface")
    print("uploading pipeline_from_organized_data_to_lentokenized_data")
    st.subheader("DNRTI to category(SecureBERT-NER) mapping:")
    st.write(dnrti_to_category)
    @st.cache_resource
    def load_pipeline(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    # Load pipelines with caching
    pipeline_from_organized_data_to_lentokenized_data = load_pipeline('pipelines/pipeline_from_organized_data_to_lentokenized_data.pkl')
    pipeline_from_predictions_to_labels = load_pipeline('pipelines/pipeline_from_predictions_to_labels.pkl')
    pipeline_from_input_to_lentokenized_data = load_pipeline('pipelines/pipeline_from_input_to_lentokenized_data.pkl')

    print("uploading model_ner")
    # model_ner
    model_ner_dir = "models/securebert_ner/model_" + str(model_ner_index) 
    model_cyner_dir = "models/securebert_cyner/model_" + str(model_cyner_index)

    model_ner = AutoModelForTokenClassification.from_pretrained(model_ner_dir)
    model_ner.eval()

    model_cyner = AutoModelForTokenClassification.from_pretrained(model_cyner_dir)
    model_cyner.eval()
    # to gpu
    # File uploader for text files
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    generate_upload_file = st.button("Generate from uploaded file")
    # add note (need to be like train.txt, test.txt, valid.txt)
    st.write("Note: The uploaded file should be in the format of the train.txt, valid.txt, or test.txt .")

    if uploaded_file and generate_upload_file:
        # Process the uploaded file
        # 
        # add generate bottom
        start_time = time.time()
        with uploaded_file as file:
            raw_data = file.read().decode("utf-8")
        # read as string
        # save it 
        # Display the modified raw data

        print("save raw_data")
        # Save the raw data
        # need to do it because of the encoder is not the same :-(
        with open('raw_data1.txt', 'w') as file:
            file.write(raw_data)
        with open('raw_data1.txt', "r", encoding="utf-8") as file:
            raw_data = file.read()  
    # delete the file
        os.remove('raw_data1.txt')
        # Check if MPS is available
        tokenized_input_dataset = pipeline_from_organized_data_to_lentokenized_data.transform(raw_data)
        predict_ner_labels = predict_pre_train_model(tokenized_input_dataset, model_ner, pipeline_from_predictions_to_labels, device=device)
        predict_cyner_labels = predict_pre_train_model(tokenized_input_dataset, model_cyner, pipeline_from_predictions_to_labels, device=device)
        # Get model_ner predictions


        reversed_map_token = {v: k for k, v in map_token.items()}    
        precision_ner, recall_ner, y_true, y_pred_ner = calculate_metrics(tokenized_input_dataset, predict_ner_labels, reversed_map_token)
        precision_cat_ner, recall_cat_ner, y_true_cat_ner, y_pred_cat_cyner = calculate_metrics(tokenized_input_dataset, predict_cyner_labels, reversed_map_token)
        # y_true_cat is the true_abels  'B-HackOrg' -> 'B-ACT' , 'O' -> 'O'
        y_true_cat = []
        y_pred_ner_cat = []
        y_pred_cyner_cat = []
        for i in range(len(y_true)):
            if y_true[i] == 'O':
                y_true_cat.append('O')
            else:
                # replace the word after the first '-' with dnrti_to_category if it is not 'O'
                if y_true[i].split('-')[1] in dnrti_to_category:
                    y_true_cat.append(y_true[i].split('-')[0] + '-' + dnrti_to_category[y_true[i].split('-')[1]])
                else:
                    y_true_cat.append('O')
            if y_pred_ner[i] == 'O':
                y_pred_ner_cat.append('O')
            else:
                if y_pred_ner[i].split('-')[1] in dnrti_to_category:
                    y_pred_ner_cat.append(y_pred_ner[i].split('-')[0] + '-' + dnrti_to_category[y_pred_ner[i].split('-')[1]])
                else:
                    y_pred_ner_cat.append('O')        
            if y_pred_cat_cyner[i] == 'O':
                y_pred_cyner_cat.append('O')
            else:
                if y_pred_cat_cyner[i].split('-')[1] in dnrti_to_category:
                    y_pred_cyner_cat.append(y_pred_cat_cyner[i].split('-')[0] + '-' + dnrti_to_category[y_pred_cat_cyner[i].split('-')[1]])
                else:
                    y_pred_cyner_cat.append('O')

        precision_ner_cat = precision_score(y_true_cat, y_pred_ner_cat, average='weighted')
        recall_ner_cat = recall_score(y_true_cat, y_pred_ner_cat, average='weighted')
        precision_cyner_cat = precision_score(y_true_cat, y_pred_cyner_cat, average='weighted')
        recall_cyner_cat = recall_score(y_true_cat, y_pred_cyner_cat, average='weighted')
        end_time = time.time()
    # Run the app with: `streamlit run ner_streamlit_app.py`
    # The app will be available at: http://localhost:8501
        st.write("DNRTI Tags NER model Precision: ", np.round(precision_ner,3))
        st.write("DNRTI Tags NER Recall: ", np.round(recall_ner,3))
        st.write("DNRTI Tags CYNER model Precision: ", np.round(precision_cat_ner,3))
        st.write("DNRTI Tags CYNER model Recall: ", np.round(recall_cat_ner,3))
        st.write("Category tags NER Precision: ", np.round(precision_ner_cat,3))
        st.write("Category tags NER Recall: ", np.round(recall_ner_cat,3))
        st.write("Category tags CYNER Precision: ", np.round(precision_cyner_cat,3))
        st.write("Category tags CYNER Recall: ", np.round(recall_cyner_cat,3))
        st.write("Latency: ", np.round(end_time - start_time, 2), "seconds")
        # generate df with colmns word and pred_label
        df = pd.DataFrame(columns=['word', 'true_DNRTI_labels', 'pred_NER_DNRTI_labels', 'pred_CyNER_DNRTI_labels', 'true_Cat_labels', 'pred_NER_Cat_labels', 'pred_CyNER_Cat_labels'])
        # words are list from input_test_example split by space and '
        print(" predict_ner_labels", predict_ner_labels)
        print("len(predict_ner_labels)", len(predict_ner_labels))
        
        for i in range(len(tokenized_input_dataset))[:5]:
            print("i", i)
            
            tokens = tokenized_input_dataset[i]['tokens']
            len_of_tokens = len(tokens)
            predict_ner_labels_list = predict_ner_labels[i][:len_of_tokens]
            predict_cyner_labels_list = predict_cyner_labels[i][:len_of_tokens]
            true_labels_list = tokenized_input_dataset[i]['labels'][:len_of_tokens]
            # map with reversed_map_token
            true_labels_list = [reversed_map_token[label] for label in true_labels_list]
            # do concat
            print("predict_cyner_labels_list", predict_cyner_labels_list)
            temp_df = pd.DataFrame({'word': tokens, 'true_DNRTI_labels': true_labels_list, 'pred_NER_DNRTI_labels': predict_ner_labels_list, 'pred_CyNER_DNRTI_labels': predict_cyner_labels_list})
            
            # make space between sentences
            temp_df['true_Cat_labels'] = temp_df['true_DNRTI_labels'].apply(
    lambda x: f"{x.split('-')[0]}-{dnrti_to_category.get(x.split('-')[1], 'O')}" if x != 'O' else 'O'
)

            temp_df['pred_NER_Cat_labels'] = temp_df['pred_NER_DNRTI_labels'].apply(
    lambda x: f"{x.split('-')[0]}-{dnrti_to_category.get(x.split('-')[1], 'O')}" if x != 'O' else 'O'
)
            temp_df['pred_CyNER_Cat_labels'] = temp_df['pred_CyNER_DNRTI_labels'].apply(
    lambda x: f"{x.split('-')[0]}-{dnrti_to_category.get(x.split('-')[1], 'O')}" if x != 'O' else 'O'
)
            df = pd.concat([df, temp_df])
            df = pd.concat([df, pd.DataFrame({'word': ['-----'], 'pred_NER_DNRTI_labels': ['----'], 'true_DNRTI_labels': ['----'], 'true_Cat_labels': ['----'], 'pred_NER_Cat_labels': ['----'], 'pred_CyNER_Cat_labels': ['----'], 'pred_CyNER_DNRTI_labels': ['----']})])
        st.subheader("Sample predictions")
        st.write(df)
    text_input = st.text_area("Input text", height=200)
    # Display the modified raw data
    text_input_generetor = st.button("Generate from text area")

    if text_input and text_input_generetor:
        tokenized_input_example = pipeline_from_input_to_lentokenized_data.transform(text_input)
        sentences = split_text_to_words_in_sentences(text_input)
        predict_labels = predict_pre_train_model(tokenized_input_example, model_ner, pipeline_from_predictions_to_labels)
        # generate df with colmns word and pred_label
        df = pd.DataFrame()
        # words are list from input_test_example split by space and '
        for i in range(len(sentences)):
            
            tokens = sentences[i]['tokens']
            len_of_tokens = len(tokens)
            predict_labels = predict_labels[i][:len_of_tokens]
            
            
            # do concat
            temp_df = pd.DataFrame({'word': tokens, 'pred_NER_DNRTI_labels': predict_labels})
            temp_df['pred_NER_Cat_labels'] = temp_df['pred_NER_DNRTI_labels'].apply(
                lambda x: f"{x.split('-')[0]}-{dnrti_to_category.get(x.split('-')[1], 'O')}" if x != 'O' else 'O'
            )
            df = pd.concat([df, temp_df])
            # make space between sentences
            df = pd.concat([df, pd.DataFrame({'word': ['-----'], 'pred_NER_DNRTI_labels': ['----'], 'pred_NER_Cat_labels': ['----']})])
        if len(df) > 1:
            st.subheader("Input Text predictions")
            st.write(df)


