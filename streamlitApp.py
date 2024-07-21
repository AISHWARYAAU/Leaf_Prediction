import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import os
import platform
import pathlib

# Platform-specific path handling
plt = platform.system()
if plt == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Set up the page layout
st.set_page_config(page_title="ChromaticScan", page_icon="🌿")

st.title("ChromaticScan")

st.caption(
    "A ResNet 34-based Algorithm for Robust Plant Disease Detection with 99.2% Accuracy Across 39 Different Classes of Plant Leaf Images."
)

st.write("Try clicking a leaf image and watch how an AI Model will detect its disease.")

with st.sidebar:
    img = Image.open("./Images/leaf.png")
    st.image(img)
    st.subheader("About ChromaticScan")
    st.write(
        "ChromaticScan is a state-of-the-art convolutional neural network (CNN) algorithm that is specifically designed for detecting plant diseases. It utilizes transfer learning by fine-tuning the ResNet 34 model on a large dataset of leaf images to achieve an impressive 99.2% accuracy in detecting various plant diseases. The algorithm is trained to identify specific patterns and features in the leaf images that are indicative of different types of diseases, such as leaf spots, blights, and wilts."
    )
    st.write(
        "ChromaticScan is designed to be highly robust and accurate, with the ability to detect plant diseases in a wide range of conditions and environments. It can be used to quickly and accurately diagnose plant diseases, allowing farmers and gardeners to take immediate action to prevent the spread of the disease and minimize crop losses. With its high level of accuracy and ease of use, ChromaticScan is poised to revolutionize the way plant diseases are detected and managed in the agricultural industry."
    )
    st.write(
        "The application will infer the one label out of 39 labels."
    )

classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
    "Background_without_leaves",
]

classes_and_descriptions = {
    "Apple___Apple_scab": "Apple with Apple scab disease detected.",
    "Apple___Black_rot": "Apple with Black rot disease detected.",
    "Apple___Cedar_apple_rust": "Apple with Cedar apple rust disease detected.",
    "Apple___healthy": "Healthy apple leaf detected.",
    "Blueberry___healthy": "Healthy blueberry leaf detected.",
    "Cherry_(including_sour)___Powdery_mildew": "Cherry with Powdery mildew disease detected.",
    "Cherry_(including_sour)___healthy": "Healthy cherry leaf detected.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn with Cercospora leaf spot or Gray leaf spot disease detected.",
    "Corn_(maize)___Common_rust_": "Corn with Common rust disease detected.",
    "Corn_(maize)___Northern_Leaf_Blight": "Corn with Northern Leaf Blight disease detected.",
    "Corn_(maize)___healthy": "Healthy corn leaf detected.",
    "Grape___Black_rot": "Grape with Black rot disease detected.",
    "Grape___Esca_(Black_Measles)": "Grape with Esca (Black Measles) disease detected.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape with Leaf blight (Isariopsis Leaf Spot) disease detected.",
    "Grape___healthy": "Healthy grape leaf detected.",
    "Orange___Haunglongbing_(Citrus_greening)": "Orange with Haunglongbing (Citrus greening) disease detected.",
    "Peach___Bacterial_spot": "Peach with Bacterial spot disease detected.",
    "Peach___healthy": "Healthy peach leaf detected.",
    "Pepper,_bell___Bacterial_spot": "Bell pepper with Bacterial spot disease detected.",
    "Pepper,_bell___healthy": "Healthy bell pepper leaf detected.",
    "Potato___Early_blight": "Potato with Early blight disease detected.",
    "Potato___Late_blight": "Potato with Late blight disease detected.",
    "Potato___healthy": "Healthy potato leaf detected.",
    "Raspberry___healthy": "Healthy raspberry leaf detected.",
    "Soybean___healthy": "Healthy soybean leaf detected.",
    "Squash___Powdery_mildew": "Squash with Powdery mildew disease detected.",
    "Strawberry___Leaf_scorch": "Strawberry with Leaf scorch disease detected.",
    "Strawberry___healthy": "Healthy strawberry leaf detected.",
    "Tomato___Bacterial_spot": "Tomato leaf with Bacterial spot disease detected.",
    "Tomato___Early_blight": "Tomato leaf with Early blight disease detected.",
    "Tomato___Late_blight": "Tomato leaf with Late blight disease detected.",
    "Tomato___Leaf_Mold": "Tomato leaf with Leaf Mold disease detected.",
    "Tomato___Septoria_leaf_spot": "Tomato leaf with Septoria leaf spot disease detected.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato leaf with Spider mites or Two-spotted spider mite disease detected.",
    "Tomato___Target_Spot": "Tomato leaf with Target Spot disease detected.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato leaf with Tomato Yellow Leaf Curl Virus disease detected.",
    "Tomato___Tomato_mosaic_virus": "Tomato leaf with Tomato mosaic virus disease detected.",
    "Tomato___healthy": "Healthy tomato leaf detected.",
    "Background_without_leaves": "No plant leaf detected in the image.",
}

# Define remedies for each class
remedies = {
    "Apple___Apple_scab": "Apply fungicide and remove affected leaves.",
    "Apple___Black_rot": "Use copper-based fungicides and prune infected branches.",
    "Apple___Cedar_apple_rust": "Use rust-resistant apple varieties and apply fungicides.",
    "Apple___healthy": "No action needed.",
    "Blueberry___healthy": "No action needed.",
    "Cherry_(including_sour)___Powdery_mildew": "Apply sulfur-based fungicides and improve air circulation.",
    "Cherry_(including_sour)___healthy": "No action needed.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Use resistant varieties and apply fungicides.",
    "Corn_(maize)___Common_rust_": "Apply fungicides and use resistant corn varieties.",
    "Corn_(maize)___Northern_Leaf_Blight": "Apply fungicides and rotate crops.",
    "Corn_(maize)___healthy": "No action needed.",
    "Grape___Black_rot": "Apply fungicides and remove infected plant parts.",
    "Grape___Esca_(Black_Measles)": "Remove infected vines and apply appropriate fungicides.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply fungicides and improve air circulation.",
    "Grape___healthy": "No action needed.",
    "Orange___Haunglongbing_(Citrus_greening)": "Remove infected trees and use disease-free planting material.",
    "Peach___Bacterial_spot": "Use copper-based bactericides and remove infected leaves.",
    "Peach___healthy": "No action needed.",
    "Pepper,_bell___Bacterial_spot": "Apply copper-based bactericides and improve air circulation.",
    "Pepper,_bell___healthy": "No action needed.",
    "Potato___Early_blight": "Apply fungicides and practice crop rotation.",
    "Potato___Late_blight": "Use resistant varieties and apply fungicides.",
    "Potato___healthy": "No action needed.",
    "Raspberry___healthy": "No action needed.",
    "Soybean___healthy": "No action needed.",
    "Squash___Powdery_mildew": "Apply sulfur-based fungicides and improve air circulation.",
    "Strawberry___Leaf_scorch": "Adjust watering practices and use appropriate fungicides.",
    "Strawberry___healthy": "No action needed.",
    "Tomato___Bacterial_spot": "Use copper-based bactericides and improve air circulation.",
    "Tomato___Early_blight": "Apply fungicides and practice crop rotation.",
    "Tomato___Late_blight": "Use resistant varieties and apply fungicides.",
    "Tomato___Leaf_Mold": "Improve air circulation and apply appropriate fungicides.",
    "Tomato___Septoria_leaf_spot": "Use resistant varieties and apply fungicides.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Apply miticides and improve irrigation practices.",
    "Tomato___Target_Spot": "Apply fungicides and improve air circulation.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Use resistant varieties and control whiteflies.",
    "Tomato___Tomato_mosaic_virus": "Use resistant varieties and manage insect vectors.",
    "Tomato___healthy": "No action needed.",
    "Background_without_leaves": "No plant leaf detected – please upload an image with a clear leaf."
}

# Function to load and preprocess the image
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Function to load the model
def load_model_file(model_path):
    if os.path.exists(model_path):
        model = load_model(model_path)
        return model
    else:
        st.error("Model file not found. Please check the path and try again.")
        return None

# Function for Plant Disease Detection
def Plant_Disease_Detection(image):
    model = load_model_file("Plant_disease.h5")
    if model is None:
        return None, None, None

    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100  # Confidence level
    return predicted_class, confidence

# Set up the sidebar
st.subheader("Select Image Input Method")
input_method = st.radio("options", ["File Uploader", "Camera Input"], label_visibility="collapsed")

# Handle image input
if input_method == "File Uploader":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        uploaded_file_img = load_image(uploaded_file)
        st.image(uploaded_file_img, caption="Uploaded Image", width=300)
        st.success("Image uploaded successfully!")

elif input_method == "Camera Input":
    st.warning("Please allow access to your camera.")
    camera_image_file = st.camera_input("Click an Image")
    if camera_image_file is not None:
        camera_file_img = load_image(camera_image_file)
        st.image(camera_file_img, caption="Camera Input Image", width=300)
        st.success("Image clicked successfully!")

# Button to trigger prediction
submit = st.button(label="Submit Leaf Image")
if submit:
    st.subheader("Output")
    if input_method == "File Uploader" and uploaded_file is not None:
        image = uploaded_file_img
    elif input_method == "Camera Input" and camera_image_file is not None:
        image = camera_file_img

    if image is not None:
        with st.spinner(text="This may take a moment..."):
            predicted_class, confidence = Plant_Disease_Detection(image)
            if predicted_class:
                st.write(f"Prediction: {predicted_class}")
                st.write(f"Description: {classes_and_descriptions.get(predicted_class, 'No description available.')}")
                st.write(f"Confidence: {confidence:.2f}%")
                
                # Prepare data for the table
                recommendation = remedies.get(predicted_class, 'No recommendation available.')
                
                data = {
                    "Details": ["Leaf Status", "Disease Name", "Recommendation", "Accuracy"],
                    "Values": ["Unhealthy" if predicted_class != "healthy" else "Healthy", 
                               predicted_class.split('___')[1] if len(predicted_class.split('___')) > 1 else 'Healthy',
                               recommendation,
                               f"{confidence:.2f}%"]
                }
                df = pd.DataFrame(data)
                st.table(df)
            else:
                st.error("Error in prediction. Please try again.")
    else:
        st.warning("Please upload or capture an image first.")
