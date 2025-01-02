import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.utils import get_custom_objects
from PIL import Image, ImageOps
import gdown
import base64

# Register the 'mae' function (Mean Absolute Error)
get_custom_objects()['mae'] = MeanAbsoluteError()

# Download model from Google Drive
def download_model_from_drive():
    try:
        file_id = '1jHc-XZ-mEQkj-l7-lVPILcqRiGrjPEg7'  # Replace with your actual file ID
        output = 'age_gender_model2.h5'
        gdown.download(f'https://drive.google.com/uc?export=download&id={file_id}', output, quiet=False)
        model = load_model(output)
        return model
    except Exception as e:
        st.error(f"Error downloading or loading the model: {str(e)}")
        return None

# Load model from Google Drive (Check if model is successfully loaded)
model = download_model_from_drive()

if model is None:
    st.error("The model could not be loaded. Please try again later.")
    st.stop()  # Stop execution if model is not available

# Gender prediction dictionary
gender_dict = {0: 'Male', 1: 'Female'}

# Image preprocessing function (Using PIL for grayscale conversion and resizing)

def preprocess_image(frame):
    # Convert PIL image to NumPy array if it's not already
    if isinstance(frame, Image.Image):
        frame = np.array(frame)
    
    # Check if the image is already grayscale (1 channel) or color (3 channels)
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frame = cv2.resize(frame, (128, 128))
    frame_array = frame / 255.0  # Normalize pixel values
    frame_array = np.expand_dims(frame_array, axis=0)  # Add batch dimension
    return frame_array


# Function to encode image to base64 (Not used but can be kept for future needs)
def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    img_bytes = buffer.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

# Function to predict age and gender
def predict(frame):
    # Preprocess the image
    img_array = preprocess_image(frame)

    # Make predictions
    pred_gender, pred_age = model.predict(img_array)

    # Gender prediction
    predicted_gender = gender_dict[round(pred_gender[0][0])]

    # Age range prediction
    age = round(pred_age[0][0])
    if age < 18:
        age_range = "Child (0-17)"
    elif age < 35:
        age_range = "Young Adult (18-34)"
    elif age < 50:
        age_range = "Adult (35-49)"
    else:
        age_range = "Senior (50+)"

    return predicted_gender, age, age_range

# Streamlit Interface
st.set_page_config(page_title="Age and Gender Prediction System", page_icon="👤", layout="centered")

# Add custom styles
st.markdown("""
    <style>
    .header {
        text-align: center;
        color: white;
        font-size: 50px;
        font-family: 'Arial', sans-serif;
    }
    .subheader {
        text-align: center;
        font-size: 20px;
        font-family: 'Arial', sans-serif;
        color: #7F8C8D;
    }
    .info-text {
        font-size: 16px;
        color: #34495E;
    }
    .button {
        background-color: #3498DB;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        border: none;
    }
    .button:hover {
        background-color: #2980B9;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.2);
    }
    .stText {
        font-family: 'Arial', sans-serif;
        font-size: 16px;
    }
    .vinay{
    color:#2980B9;
    }
    </style>
    """, unsafe_allow_html=True)

# Project Info
st.markdown("<div class='header'>Age and Gender Prediction System</div>", unsafe_allow_html=True)
st.markdown("""
    <div class='subheader'>
    This app uses a deep learning model developed by <strong class = 'vinay'>Vinay Karthik Kumanduri</strong>, 
    <a href="https://vinaykarthik.netlify.app" target="_blank" style="color: #3498db; font-weight: bold;">Approach Me</a><br>
    It predicts the age and gender of a person based on their image. You can either upload an image or use the webcam to capture one and get predictions.
</div>

""", unsafe_allow_html=True)

st.markdown("""
    <div class='info-text'>
        <strong>How it works:</strong><br>
        1. Upload an image or use the webcam to capture an image.<br>
        2. The model predicts the age and gender of the person in the image.<br>
        3. The results are displayed, including the predicted gender, age, and age range.<br>
        4. Make sure that your face gets enough brightness.<br><br>
        This is powered by a trained deep learning model using Convolutional Neural Networks (CNN).
    </div>
""", unsafe_allow_html=True)

# Choose between webcam or image upload
mode = st.radio("Select input mode", ("Upload Image", "Use Webcam"))

if mode == "Use Webcam":
    # Use Streamlit's camera input feature
    image = st.camera_input("Take a picture")

    if image is not None:
        # Process and predict
        img = Image.open(image)
        img = np.array(img)
        predicted_gender, age, age_range = predict(img)
        st.image(img, caption="Captured Image", use_container_width=True)
        st.write(f"**Predicted Gender:** {predicted_gender}")
        st.write(f"**Predicted Age Range:** {age_range}")
        st.write(f"**Predicted Age:** {age}")
    else:
        st.warning("Please enable your webcam to take a picture.")

elif mode == "Upload Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Open and process the uploaded image
        img = Image.open(uploaded_image)
        img = np.array(img)
        predicted_gender, age, age_range = predict(img)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        st.write(f"**Predicted Gender:** {predicted_gender}")
        st.write(f"**Predicted Age Range:** {age_range}")
        st.write(f"**Predicted Age:** {age}")
