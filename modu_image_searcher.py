import streamlit as st
import os
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Set page configuration
st.set_page_config(
    page_title="Modu: Image Searcher",
    page_icon="🔍",
    layout="wide",
)

# Inject CSS for background image
st.markdown(
    """
    <style>
    body {
        background-image: url('logo.png');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .main {
        background-color: rgba(0, 0, 0, 0.7); /* Dark overlay for readability */
        color: white;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the MobileNetV2 model
model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# Helper function to preprocess images
def preprocess_image(image):
    """Preprocess an image to feed into the MobileNetV2 model."""
    image = image.resize((224, 224))  # Resize to model input size
    image_array = np.array(image)
    return preprocess_input(np.expand_dims(image_array, axis=0))

# Helper function to compute similarity
def compute_similarity(query_features, dataset_features):
    """Compute cosine similarity between the query image and dataset images."""
    query_norm = np.linalg.norm(query_features)
    similarities = [
        np.dot(query_features, features) / (query_norm * np.linalg.norm(features))
        for features in dataset_features
    ]
    return similarities

# App title and description
st.title("🔍 Modu: Image Searcher")
st.subheader("Effortlessly search your image library with AI-powered tools")

# Dataset uploader
uploaded_dataset_files = st.file_uploader(
    "Upload your image dataset (jpg, png, jpeg):", type=["jpg", "png", "jpeg"], accept_multiple_files=True
)

# File uploader for query image
uploaded_query_file = st.file_uploader("Upload an image to search for similar images:", type=["jpg", "png", "jpeg"])

# Keyword input
keywords = st.text_input("Enter keywords (comma-separated):")

# Search button
if st.button("Start Search"):
    if not uploaded_dataset_files:
        st.error("Please upload a dataset of images.")
    else:
        dataset_features = []
        dataset_images = []
        for uploaded_file in uploaded_dataset_files:
            image = Image.open(uploaded_file)
            dataset_images.append((uploaded_file.name, image))

            # Extract features
            preprocessed_image = preprocess_image(image)
            features = model.predict(preprocessed_image)[0]
            dataset_features.append(features)

        results_found = False

        # Perform keyword search
        if keywords:
            keyword_list = [kw.strip().lower() for kw in keywords.split(",")]
            st.write(f"Searching for: {keyword_list}")
            for name, image in dataset_images:
                preprocessed_image = preprocess_image(image)
                preds = model.predict(preprocessed_image)
                decoded_preds = decode_predictions(preds, top=5)[0]
                if any(keyword in label.lower() for keyword in keyword_list for _, label, _ in decoded_preds):
                    results_found = True
                    st.image(image, caption=f"{name} (Matched Keyword)", use_column_width=True)

        # Perform image-based search
        if uploaded_query_file:
            query_image = Image.open(uploaded_query_file)
            st.image(query_image, caption="Query Image", use_column_width=True)

            # Extract features for the query image
            query_features = model.predict(preprocess_image(query_image))[0]

            # Compute similarity
            similarities = compute_similarity(query_features, dataset_features)

            # Rank results by similarity
            ranked_results = sorted(
                zip(dataset_images, similarities), key=lambda x: x[1], reverse=True
            )

            # Display results
            st.subheader("Similar Images:")
            for (name, image), similarity in ranked_results[:5]:  # Display top 5 results
                results_found = True
                st.image(image, caption=f"{name} (Similarity: {similarity:.2f})", use_column_width=True)

        if not results_found:
            st.warning("No matches found.")

# Footer
st.markdown(
    """
    ---
    <div style="text-align: center;">
        <p style="font-size: small;">© 2024 Modu: Image Searcher | <a href="https://moduimagesearch.com" target="_blank">moduimagesearch.com</a></p>
    </div>
    """,
    unsafe_allow_html=True,
)
