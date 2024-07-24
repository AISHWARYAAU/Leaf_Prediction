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

# Classes and descriptions
classes = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy", "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch",
    "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy", "Background_without_leaves"
]

classes_and_descriptions = {
    "Apple___Apple_scab": "Apple with Apple scab disease detected.",
    "Apple___Black_rot": "Apple with Black rot disease detected.",
    "Apple___Cedar_apple_rust": "Apple with Cedar apple rust disease detected.",
    "Apple___healthy": "Healthy apple leaf detected.",
    # Add other class descriptions similarly...
    "Background_without_leaves": "No plant leaf detected in the image.",
}

remedies = {
    "Apple___Apple_scab": "Apply fungicide and remove affected leaves.",
    "Apple___Black_rot": "Use copper-based fungicides and prune infected branches.",
    "Apple___Cedar_apple_rust": "Use rust-resistant apple varieties and apply fungicides.",
    "Apple___healthy": "No action needed.",
    # Add other remedies similarly...
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

# Home Page
if page == "Home":
    st.write(
        "Welcome to ChromaticScan, your AI-powered solution for detecting plant diseases. "
        "Use the sidebar to navigate to the prediction or charts sections."
    )
    st.subheader("Benefits of Plant Disease Prediction")
    st.write("""
    - **Early Detection**: Identifying diseases at an early stage helps in timely intervention, preventing widespread damage.
    - **Cost-Effective**: Early treatment reduces the need for extensive use of pesticides and other treatments, saving costs.
    - **Increased Yield**: Healthy plants result in better yield and quality, ensuring profitability for farmers.
    - **Data-Driven Decisions**: Use of AI and machine learning provides insights that can guide agricultural practices and strategies.
    """)

    st.subheader("Usage")
    st.write("""
    - **Upload or capture a leaf image**: Use the app to upload an image of a plant leaf or take a picture using the camera.
    - **Receive diagnosis and recommendations**: The app will predict the disease and provide recommendations for treatment.
    - **Monitor and manage**: Regular use of the app can help in monitoring plant health and managing diseases effectively.
    """)

    st.subheader("Machine Learning Algorithm")
    st.write("""
    - **ResNet 34**: ChromaticScan uses a deep learning model based on ResNet 34, a type of convolutional neural network.
    - **Transfer Learning**: The model is fine-tuned using a dataset of plant leaf images, leveraging pre-trained weights for improved accuracy.
    - **High Accuracy**: The model achieves an accuracy of 99.2%, capable of distinguishing between 39 different classes of plant diseases and healthy leaves.
    """)

# Prediction Page
elif page == "Prediction":
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


# Charts Page
elif page == "Charts":
    st.subheader("Charts and Visualizations")

    # Sample data for accuracy and loss
    # Replace these with your actual data
    epochs = range(1, 21)
    accuracy = [0.7, 0.75, 0.78, 0.82, 0.85, 0.87, 0.88, 0.89, 0.9, 0.91, 0.91, 0.92, 0.92, 0.93, 0.93, 0.94, 0.94, 0.95, 0.95, 0.95]
    val_accuracy = [0.68, 0.72, 0.75, 0.8, 0.83, 0.85, 0.86, 0.87, 0.88, 0.89, 0.89, 0.9, 0.91, 0.91, 0.92, 0.92, 0.93, 0.93, 0.93, 0.94]
    loss = [0.5, 0.45, 0.4, 0.35, 0.3, 0.28, 0.25, 0.23, 0.2, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08]
    val_loss = [0.52, 0.48, 0.42, 0.38, 0.34, 0.31, 0.29, 0.27, 0.25, 0.23, 0.21, 0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12]

    st.write("### Accuracy over Epochs")
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, accuracy, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    st.write("### Loss over Epochs")
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Sample confusion matrix
    # Replace with your actual confusion matrix data
    conf_matrix = np.random.randint(0, 100, size=(5, 5))

    st.write("### Confusion Matrix")
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)

    # Sample class distribution
    # Replace with your actual class distribution data
    class_distribution = {
        "Apple": 200, "Blueberry": 150, "Cherry": 180, "Corn": 220, "Grape": 240,
        "Orange": 210, "Peach": 190, "Pepper": 230, "Potato": 250, "Tomato": 300
    }
    
    st.write("### Class Distribution")
    plt.figure(figsize=(10, 5))
    plt.bar(class_distribution.keys(), class_distribution.values())
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    st.pyplot(plt)
