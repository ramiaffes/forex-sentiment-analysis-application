import streamlit as st
from transformers import XLNetForSequenceClassification, XLNetTokenizer
import torch

# Cache the model and tokenizer to avoid reloading
@st.cache_resource
def load_model():
    model_path = "C:\\Users\\Admin\\forex-sentiment-analysis\\models\\forex_analysis_model"
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-large-cased")
    model = XLNetForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set model to evaluation mode
    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model()

# Define label mapping
label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Streamlit app UI
st.title("📈 Forex Sentiment Analysis")
st.write("Analyze the sentiment of forex-related headlines.")

# Example headlines
st.write("**Examples:**")
st.write("- USD surges as inflation fears grow")
st.write("- Euro falls amid political uncertainty")

# Input text box
user_input = st.text_input("Enter a headline:", "", placeholder="e.g., USD surges as inflation fears grow")

# Predict button
if st.button("Analyze Sentiment"):
    user_input = user_input.strip()  # Remove leading/trailing spaces
    if user_input:
        with st.spinner("Analyzing..."):
            try:
                # Tokenize input
                inputs = tokenizer(user_input, return_tensors="pt", padding="max_length", truncation=True, max_length=46)

                # Make prediction
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    predicted_label = torch.argmax(logits, dim=1).item()
                    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()

                # Display result
                sentiment = label_mapping.get(predicted_label, "Unknown")
                st.success(f"**Sentiment:** {sentiment}")
                st.write(f"**Confidence:** {probabilities[predicted_label]:.2f}")
            except Exception as e:
                st.error(f"Error processing input: {str(e)}")
    else:
        st.warning("Please enter a valid headline.")
