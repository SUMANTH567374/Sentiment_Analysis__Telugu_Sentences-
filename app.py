import re
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import joblib
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertModel.from_pretrained("bert-base-multilingual-cased")

# Load the sentiment classification model (best_model.pkl)
best_model = joblib.load(r'C:\Users\LENOVO\Downloads\Telugu_sentiment_Analysis\Sentiment_Analysis__Telugu_Sentences-\best_model.pkl')

# Initialize and fit the LabelEncoder with the same labels used during training
label_encoder = LabelEncoder()
label_mapping = {0: "neg", 1: "neutral", 2: "pos"}  # Ensure the mapping matches the training process
label_encoder.fit(list(label_mapping.keys()))  # Fit encoder with numeric labels (0, 1, 2)

# Function to preprocess Telugu text
def preprocess_text(text):
    # Remove special characters, numbers, and punctuations (retain Telugu characters and spaces)
    text = re.sub(r"[^\u0C00-\u0C7F\s]", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Tokenize and encode the text using BERT
def encode_text(text):
    # Tokenize and get embeddings from the BERT model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings

# Streamlit UI
st.set_page_config(page_title="Telugu Sentiment Analysis", page_icon="📝", layout="wide")

# Sidebar information
st.sidebar.header("How it Works:")
st.sidebar.markdown(
    """
    1. **Enter** a Telugu sentence in the text box below.
    2. The app processes your sentence using **BERT**, a powerful language model that understands Telugu.
    3. Based on the input, the model predicts the sentiment as one of the following:
        - **Positive** 😊
        - **Negative** 😞
        - **Neutral** 😐
    4. Press the **Predict Sentiment** button to get the result!
    """
)

# Stylish header with emojis and gradient
st.markdown(
    """
    <h1 style="text-align: center; color: #FF6347; font-family: 'Arial'; font-size: 50px;">🌟 Telugu Sentiment Analysis 🌟</h1>
    <p style="text-align: center; color: #6C757D; font-size: 20px;">Enter a Telugu sentence below to predict its sentiment.</p>
    """, unsafe_allow_html=True
)

# Input area with border, rounded corners, and placeholder
user_input = st.text_area("✍️ Enter Telugu Sentence", "", height=150, max_chars=300, key="user_input", 
                          help="Type a sentence in Telugu to check its sentiment.")

# Prediction button with custom styling
prediction_button = st.button("🔮 Predict Sentiment", help="Click to predict the sentiment of the entered text")

# Loading spinner
with st.spinner('Predicting sentiment...'):
    if prediction_button and user_input:
        # Preprocess and tokenize the input
        cleaned_text = preprocess_text(user_input)
        
        # Get embeddings from BERT
        input_embedding = encode_text(cleaned_text)
        
        # Reshape for prediction
        input_embedding = input_embedding.reshape(1, -1)
        
        # Make prediction
        sentiment_pred = best_model.predict(input_embedding)
        
        # Handle unexpected labels
        if sentiment_pred[0] in label_mapping:
            sentiment = label_mapping[sentiment_pred[0]]
        else:
            st.error("🚨 Unexpected label encountered in prediction! 🚨")
            sentiment = None
        
        # Display result with interactive styling
        if sentiment == "pos":
            st.success(f"✨ **Predicted Sentiment**: **Positive** 😊")
        elif sentiment == "neg":
            st.error(f"😞 **Predicted Sentiment**: **Negative** 😞")
        elif sentiment == "neutral":
            st.info(f"😐 **Predicted Sentiment**: **Neutral** 😐")
        else:
            st.warning("🚨 Something went wrong! Please try again. 🚨")

# Footer with custom design and HTML for styling
st.markdown("---")
st.markdown(
    """
    <p style="text-align: center; font-size: 18px; color: #6C757D;">
        ✨ Made with ❤️ by Sumanth ✨
    </p>
    <p style="text-align: center; font-size: 16px; color: #6C757D;">
        This app uses BERT for multilingual sentiment analysis, specifically for Telugu text processing. 
    </p>
    """,
    unsafe_allow_html=True,
)







