import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

# ----------------------------
# Load model and tokenizer (corrected)
# ----------------------------
@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model("lstm_model_smote.keras")  # or .h5 if you saved in HDF5
    with open("tokenizer.json", "r") as f:
        tokenizer_json = f.read()  # Read as string
    tokenizer = tokenizer_from_json(tokenizer_json)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# ----------------------------
# Constants (match training)
# ----------------------------
MAX_SEQUENCE_LENGTH = 100
labels = ['NO', 'YES']  # 0 = NO depression, 1 = YES depression

# ----------------------------
# UI Layout
# ----------------------------
st.title("🧠 Depression Detection from Text")
st.write("This LSTM-based model predicts if the input text shows signs of depression.")

# Text input
user_input = st.text_area("💬 Enter your text here", height=150)

# Predict button
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        # Tokenize and pad
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

        # Predict
        prediction = model.predict(padded)
        pred_class = int(np.argmax(prediction))
        confidence = float(prediction[0][pred_class])

        # Result
        st.subheader("🧾 Prediction")
        st.success(f"Prediction: **{labels[pred_class]}** (confidence: {confidence:.2f})")