import streamlit as st
import os
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Modu: Image Searcher",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject CSS for background image and styling
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
        background-color: rgba(0, 0, 0, 0.7); /* Semi-transparent overlay */
        color: white;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display app header
st.title("🔍 Modu: Image Searcher")
st.subheader("Effortlessly search your image library with AI-powered tools")

# Load the MobileNetV2 model
model = MobileNetV2(weights="imagenet")

# Helper function to preprocess images
def preprocess_image(image_path):
    """Preprocess an image to feed into the MobileNetV2 model."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image)
    return preprocess_input(np.expand_dims(image_array, axis=0))

# Function to search images by keywords
def search_images(folder, keywords):
    """Search for images matching the keywords."""
    results = []
    keywords = [kw.strip().lower() for kw in keywords]
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                try:
                    preprocessed = preprocess_image(file_path)
                    preds = model.predict(preprocessed)
                    decoded = decode_predictions(preds, top=5)[0]
                    for _, label, _ in decoded:
                        if any(keyword in label.lower() for keyword in keywords):
                            results.append(file_path)
                            break
                except Exception as e:
                    st.error(f"Error processing {file}: {e}")
    return results

# Input fields for folder and keywords
folder = st.text_input("Enter folder path")
keywords = st.text_input("Enter keywords (comma-separated)")

# Search button
if st.button("Search"):
    if not folder or not keywords:
        st.warning("Please provide a folder path and keywords.")
    else:
        keyword_list = keywords.split(",")
        st.write(f"Searching for keywords: {keyword_list}")
        results = search_images(folder, keyword_list)
        if results:
            st.success(f"Found {len(results)} matching images!")
            for img_path in results:
                st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
        else:
            st.warning("No matching images found.")

# Footer with branding
st.markdown(
    """
    ---
    <div style="text-align: center;">
        <p style="font-size: small;">© 2024 Modu: Image Searcher | <a href="https://moduimagesearch.com" target="_blank">moduimagesearch.com</a></p>
    </div>
    """,
    unsafe_allow_html=True,
)
