# app.py
import streamlit as st
import joblib
from transformers import BertTokenizer, TFBertForSequenceClassification


model = TFBertForSequenceClassification.from_pretrained("sentiment_analysis_model")
tokenizer = BertTokenizer.from_pretrained("sentiment_analysis_model")

# واجهة المستخدم
st.title("Sentiment Analysis Classifier")
text_input = st.text_area("Enter text to classify:")

if st.button("Classify"):
    if text_input:
        # معالجة النص
        text_vector = vectorizer.transform([text_input])
        prediction = model.predict(text_vector)[0]
        st.success(f"Predicted Sentiment: {prediction}")
    else:
        st.warning("Please enter some text first.")
