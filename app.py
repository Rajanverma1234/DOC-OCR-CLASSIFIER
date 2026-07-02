"""
app.py

Streamlit demo: upload a document image, see the cleaned version,
predicted class, and extracted text.

Run with:  streamlit run app.py
"""

import tempfile
import os

import streamlit as st
from PIL import Image

from src.predict import process_document
from src.preprocess import preprocess_image

st.set_page_config(page_title="Document OCR & Classifier", layout="centered")

st.title("Deep Learning Document OCR & Classification")
st.write("Upload a scanned document (invoice, resume, ID card, receipt, etc.) to classify it and extract its text.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "tiff"])

if uploaded_file is not None:
    # Save to a temp file since our pipeline works off file paths
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(Image.open(tmp_path), use_container_width=True)

    with col2:
        st.subheader("Cleaned (preprocessed)")
        cleaned = preprocess_image(tmp_path)
        st.image(cleaned, use_container_width=True, clamp=True)

    with st.spinner("Classifying and extracting text..."):
        try:
            result = process_document(tmp_path)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

    st.subheader("Prediction")
    st.write(f"**Document type:** {result['predicted_class']}  ({result['classification_confidence']}% confidence)")
    st.bar_chart(result["class_probabilities"])

    st.subheader("Extracted Text")
    st.write(f"OCR confidence: {result['ocr_avg_confidence']}%")
    st.text_area("Text", result["ocr_text"], height=200)

    os.unlink(tmp_path)
