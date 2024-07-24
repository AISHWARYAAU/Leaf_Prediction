import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import os
import platform
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns

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

# Sidebar for navigation
with st.sidebar:
    img = Image.open("./Images/leaf.png")
    st.image(img)
    st.subheader("Navigation")
    page = st.radio("Go to", ["Home", "Prediction", "Charts"])

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

# Home Page
if page == "Home":
    st.write(
        "Welcome to ChromaticScan, your AI-powered solution for detecting plant diseases. "
        "Use the sidebar to navigate to the prediction or charts sections."
    )

# Prediction Page
elif page == "Prediction":
    st.subheader("Upload an Image for Prediction")

    input_method = st.radio("Select Image Input Method", ["File Uploader", "Camera Input"], label_visibility="collapsed")

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

# Charts Page
elif page == "Charts":
    st.subheader("Charts and Visualizations")

    # Sample data for accuracy and loss
    # Replace these with your actual data
    epochs = range(1, 21)
    accuracy = [0.7, 0.75, 0.78, 0.82, 0.85, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.92, 0.93, 0.94, 0.94, 0.95, 0.95, 0.96, 0.96, 0.96]
    val_accuracy = [0.68, 0.74, 0.76, 0.81, 0.83, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.91, 0.92, 0.93, 0.93, 0.94, 0.94, 0.95, 0.95, 0.95]

    loss = [0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.28, 0.26, 0.24, 0.22, 0.2, 0.18, 0.16, 0.14, 0.12, 0.1, 0.08, 0.06, 0.04]
    val_loss = [0.62, 0.58, 0.53, 0.48, 0.43, 0.38, 0.34, 0.32, 0.3, 0.28, 0.26, 0.24, 0.22, 0.2, 0.18, 0.16, 0.14, 0.12, 0.1, 0.08]

    # Plot Accuracy and Loss
    st.write("### Training and Validation Accuracy")
    fig, ax = plt.subplots()
    ax.plot(epochs, accuracy, label='Training Accuracy')
    ax.plot(epochs, val_accuracy, label='Validation Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    st.pyplot(fig)

    st.write("### Training and Validation Loss")
    fig, ax = plt.subplots()
    ax.plot(epochs, loss, label='Training Loss')
    ax.plot(epochs, val_loss, label='Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    st.pyplot(fig)

    # Confusion Matrix
    # Assuming `confusion_matrix` is available
    # Replace this with your actual confusion matrix data
    confusion_matrix = np.array([
        [80, 2, 1, 0, 0],
        [3, 76, 2, 1, 1],
        [2, 2, 82, 1, 1],
        [0, 1, 2, 79, 2],
        [0, 0, 1, 2, 81]
    ])

    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    # Class Distribution (Sample data)
    # Replace these with your actual class distribution data
    classes = ["Apple_scab", "Black_rot", "Cedar_apple_rust", "Healthy", "Powdery_mildew"]
    num_images = [100, 150, 120, 130, 110]

    st.write("### Class Distribution")
    fig, ax = plt.subplots()
    ax.bar(classes, num_images)
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Images')
    st.pyplot(fig)
