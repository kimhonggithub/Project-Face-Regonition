import streamlit as st
import pickle
import cv2
import numpy as np
from skimage.transform import resize

# Load the model and PCA object
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Assuming you have the same pre-processing steps as in your training script
def resize_image(image, target_size=(100, 100)):
    resized_image = cv2.resize(np.array(image), target_size)
    return resized_image

def gray_scale(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    return gray_image

# Define the face prediction function
def predict_face(image, clf, pca, scaler):
    # Resize the image to the target size
    resized_image = resize_image(image, target_size=(80, 80))

    # Convert the image to grayscale
    gray_image = gray_scale(resized_image)

    # Flatten and scale the image
    flattened_image = gray_image.reshape(1, -1)
    scaled_image = scaler.transform(flattened_image)

    # Apply PCA transformation
    transformed_image = pca.transform(scaled_image)

    # Make prediction
    prediction = clf.predict(transformed_image)[0]

    return prediction

# Streamlit app
def main():
    st.title("Face Recognition App")

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=200)
        # Make prediction using the loaded model, PCA object, and scaler
        prediction = predict_face(image, model, pca, scaler)
        
        st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()
