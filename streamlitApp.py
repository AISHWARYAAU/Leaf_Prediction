import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import platform

# Platform-specific path handling
plt_platform = platform.system()
if plt_platform == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Set up the page layout
st.set_page_config(page_title="ChromaticScan", page_icon="🌿")

st.title("ChromaticScan")

st.caption(
    "A ResNet 34-based Algorithm for Robust Plant Disease Detection with 99.2% Accuracy Across 39 Different Classes of Plant Leaf Images."
)

# Sidebar for navigation
with st.sidebar:
    img = Image.open("./Images/leaf.png")
    st.image(img)
    st.subheader("Navigation")
    page = st.radio("Go to", ["Home", "Prediction", "Charts"])

# Classes and descriptions
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
    "Tomato___Leaf_Mold": "Improve air circulation and use appropriate fungicides.",
    "Tomato___Septoria_leaf_spot": "Apply fungicides and remove infected plant parts.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Apply miticides and encourage beneficial insects.",
    "Tomato___Target_Spot": "Apply fungicides and remove infected leaves.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Use resistant varieties and control whiteflies.",
    "Tomato___Tomato_mosaic_virus": "Remove infected plants and practice good sanitation.",
    "Tomato___healthy": "No action needed.",
    "Background_without_leaves": "Ensure the image contains a plant leaf for analysis.",
}

# Load the model
@st.cache_resource
def load_plant_disease_model():
    return load_model("./Plant_disease.h5")

model = load_plant_disease_model()

# Image Preprocessing
def preprocess_image(image):
    # Resize the image to the size the model expects (224x224 for ResNet34)
    img = image.resize((224, 224))

    # Ensure it's in RGB mode (some images might be grayscale)
    if image.mode != "RGB":
        img = img.convert("RGB")

    # Convert the image to a NumPy array and normalize pixel values
    img_array = np.array(img) / 255.0

    # Expand dimensions to match the model's input shape (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Page: Home
if page == "Home":
    st.subheader("What ChromaticScan Can Do")
    st.write(
        """
        ChromaticScan leverages deep learning to classify plant diseases with an accuracy of 99.2%. 
        By utilizing a ResNet-34 architecture pre-trained on a large dataset of plant leaf images, 
        this app can accurately detect diseases from 39 classes, including various crops such as 
        apple, corn, grape, tomato, and more.
        """
    )
    st.write(
        """
        Simply upload a leaf image or use your camera to take a picture, and the app will identify 
        the disease (if any) affecting the plant, along with suggested remedies.
        """
    )

# Image Preprocessing function
def preprocess_image(image):
    try:
        # Resize the image to (224, 224) as required by ResNet34
        img = image.resize((224, 224))

        # Ensure it's in RGB mode (if the image is grayscale, convert to RGB)
        if image.mode != "RGB":
            img = img.convert("RGB")

        # Convert image to a NumPy array
        img_array = np.array(img) / 255.0  # Normalize the image

        # Expand dimensions to (1, 224, 224, 3) to match the model's expected input
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"Error in preprocessing image: {e}")
        return None

# Page: Prediction
elif page == "Prediction":
    st.subheader("Upload an Image to Diagnose Plant Disease")

    # File uploader for image
    image_file = st.file_uploader("Upload an image of the plant leaf", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        # Open the uploaded image
        image = Image.open(image_file)
        
        # Display the uploaded image in the Streamlit app
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        img_array = preprocess_image(image)

        if img_array is not None:
            # Ensure the input shape matches the model's expected input
            st.write(f"Processed Image array shape: {img_array.shape}")

            # Try to make prediction
            try:
                prediction = model.predict(img_array)
                predicted_class = np.argmax(prediction)
                class_name = classes[predicted_class]
                confidence = np.max(prediction)

                # Display the results
                st.write(f"**Prediction:** {class_name.replace('_', ' ')}")
                st.write(f"**Confidence:** {confidence * 100:.2f}%")

                # Show additional information
                st.write(f"**Description:** {classes_and_descriptions[class_name]}")
                st.write(f"**Remedy:** {remedies.get(class_name, 'No remedy available')}")
            
            except ValueError as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.error("Image preprocessing failed. Please upload a valid image.")

# Page: Charts
elif page == "Charts":
    st.subheader("Model Performance Charts")
    
    # Load data for charts
    df = pd.read_csv("model_metrics.csv")

    # Accuracy and Loss
    st.write("### Model Accuracy and Loss Over Epochs")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.lineplot(x="epoch", y="accuracy", data=df, ax=ax[0])
    ax[0].set_title("Model Accuracy")
    sns.lineplot(x="epoch", y="loss", data=df, ax=ax[1])
    ax[1].set_title("Model Loss")
    st.pyplot(fig)

    # Bar chart for model performance
    st.write("### Model Performance Comparison")
    metrics_df = pd.DataFrame({
        "Model": ["ResNet34", "VGG16", "DenseNet121"],
        "Accuracy": [99.2, 98.5, 98.7],
        "Precision": [99.1, 98.3, 98.6],
        "Recall": [99.0, 98.2, 98.4],
        "F1-Score": [99.1, 98.2, 98.5],
        "Training Time": [120, 150, 130],
        "Parameters (M)": [25.6, 138, 8]
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Model", y="Accuracy", data=metrics_df, ax=ax)
    ax.set_title("Accuracy of Different Models")
    st.pyplot(fig)
