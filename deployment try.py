import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pickle

# Load the trained model and binarizer
model = DistilBertForSequenceClassification.from_pretrained('distilbert-finetuned-imdb-multi-label')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

with open('multi-label-binarizer.pkl', 'rb') as f:
    multilabel_binarizer = pickle.load(f)

# Streamlit app
st.title("Movie Genre Classification")

# User input
description = st.text_area("Enter movie description:")

if st.button("Predict Genre"):
    if description:
        # Tokenize input
        inputs = tokenizer(description, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Apply sigmoid and threshold
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(outputs.logits.squeeze())
        predictions = (probs >= 0.3).float().cpu().numpy()

        # Convert predictions to genre labels
        genres = multilabel_binarizer.inverse_transform([predictions])
        
        st.write(f"Predicted Genres: {genres[0]}")
    else:
        st.write("Please enter a movie description.")
