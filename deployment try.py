# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 20:49:30 2024

@author: HP
"""

# app.py

import streamlit as st
import torch
import pickle
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load the MultiLabelBinarizer and Model
@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-finetuned-imdb-multi-label")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-finetuned-imdb-multi-label")
    with open("multi-label-binarizer.pkl", "rb") as f:
        multilabel = pickle.load(f)
    return tokenizer, model, multilabel

tokenizer, model, multilabel = load_model_and_tokenizer()

# Title for the Streamlit App
st.title("IMDB Multi-Label Movie Genre Prediction")

# Text input field for prediction
text = st.text_area("Enter movie description for genre prediction:")

if st.button("Predict"):
    if text:
        # Tokenize and prepare the text for the model
        encoding = tokenizer(text, return_tensors='pt')
        encoding.to(model.device)

        # Get model predictions
        with torch.no_grad():
            outputs = model(**encoding)
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(outputs.logits[0].cpu())
            preds = np.zeros(probs.shape)
            preds[np.where(probs >= 0.3)] = 1

        # Inverse transform to get the predicted genres
        predicted_genres = multilabel.inverse_transform(preds.reshape(1, -1))

        # Display the results
        if predicted_genres:
            st.success(f"Predicted Genres: {', '.join(predicted_genres[0])}")
        else:
            st.warning("No genres predicted with the current threshold.")
    else:
        st.error("Please enter a movie description.")

# Download the trained model and binarizer
st.markdown("### Download the Model and MultiLabelBinarizer:")
st.markdown('[Download MultiLabelBinarizer](multi-label-binarizer.pkl)')
st.markdown('[Download Trained Model](distilbert-finetuned-imdb-multi-label.zip)')
