import os

def load_model_file(model_path):
    # Print the current working directory
    st.write(f"Current working directory: {os.getcwd()}")

    # Print the absolute path of the model file
    absolute_path = os.path.abspath(model_path)
    st.write(f"Model file absolute path: {absolute_path}")

    if os.path.exists(model_path):
        model = load_model(model_path)
        return model
    else:
        st.error(f"Model file not found at {absolute_path}. Please check the path and try again.")
        return None

# Usage in Plant_Disease_Detection
def Plant_Disease_Detection(image_path):
    model = load_model_file("Plant_disease.h5")  # Adjust the path if needed
    if model is None:
        return None, None

    image = Image.open(image_path).resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class = classes[np.argmax(prediction)]
    return prediction, predicted_class
