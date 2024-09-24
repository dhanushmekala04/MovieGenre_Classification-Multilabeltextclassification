
import streamlit as st
import pickle
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# Load model, tokenizer, and binarizer
checkpoint = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(checkpoint)
model = DistilBertForSequenceClassification.from_pretrained(checkpoint, 
                                                            problem_type="multi_label_classification")

# Load the binarizer
multilabel = pickle.load(open("multilabel_binarizer.pkl", "rb"))

# Define function to make predictions
def predict(texts):
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt', max_length=128)
    inputs = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids=inputs, attention_mask=attention_mask)
    
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(outputs.logits).numpy()
    
    # Use a threshold to convert probabilities to binary predictions
    threshold = 0.3
    preds = (probs >= threshold).astype(int)
    
    # Convert predictions to labels
    predicted_labels = multilabel.inverse_transform(preds)
    
    return predicted_labels

# Streamlit app UI
st.title("DistilBERT Multi-Label Classification")

# User input
text_input = st.text_area("Enter text to classify:")

if st.button("Classify"):
    if text_input:
        predictions = predict([text_input])
        st.write("Predicted Labels:", predictions[0])
    else:
        st.error("Please enter some text.")
