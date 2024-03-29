import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler

# Load the CountVectorizer and MinMaxScaler
cv = pickle.load(open('countVectorizer.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Load the trained Random Forest model
model_rf = pickle.load(open('model_rf.pkl', 'rb'))

# Function to preprocess text input
def preprocess_text(text):
    # Tokenize and transform text using CountVectorizer
    text_cv = cv.transform([text])
    # Convert sparse matrix to dense numpy array
    text_cv_dense = text_cv.toarray()
    # Scale the text features using MinMaxScaler
    text_scl = scaler.transform(text_cv_dense)
    return text_scl

# Function to predict sentiment
def predict_sentiment(text):
    # Preprocess the text input
    text_scl = preprocess_text(text)
    # Predict sentiment using the trained model
    prediction = model_rf.predict(text_scl)
    return prediction[0]

# Streamlit UI
def main():
    st.title("Text Sentiment Predictor")
    st.sidebar.title("Options")

    option = st.sidebar.selectbox("Choose an option", ["Single Text Prediction"])

    if option == "Single Text Prediction":
        st.subheader("Enter Text")
        text_input = st.text_area("Enter your text here", "")
        if st.button("Predict"):
            if text_input.strip() != "":
                # Predict sentiment
                prediction = predict_sentiment(text_input)
                if prediction == 1:
                    st.write("Sentiment: Positive")
                else:
                    st.write("Sentiment: Negative")
            else:
                st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
