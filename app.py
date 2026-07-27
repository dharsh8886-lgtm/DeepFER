from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(
    page_title="DeepFER",
    page_icon="😊",
    layout="centered",
)

MODEL_PATH = Path("models/deepfer_model.keras")
CLASS_NAMES_PATH = Path("models/class_names.txt")


@st.cache_resource
def load_emotion_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return tf.keras.models.load_model(MODEL_PATH)


def load_class_names():
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            names = [line.strip() for line in file if line.strip()]

        if names:
            return names

    return [
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    ]


def prepare_image(image: Image.Image) -> np.ndarray:
    # Correct image rotation using EXIF information
    image = ImageOps.exif_transpose(image)

    # Convert to grayscale, matching model training
    image = image.convert("L")

    # Resize to the CNN input size
    image = image.resize((48, 48))

    image_array = np.asarray(image, dtype=np.float32)

    # Normalize exactly like the training dataset
    image_array = image_array / 255.0

    # Shape: (1, 48, 48, 1)
    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_emotion(image: Image.Image, model, class_names):
    input_array = prepare_image(image)

    prediction = model.predict(input_array, verbose=0)[0]

    emotion_index = int(np.argmax(prediction))
    confidence = float(prediction[emotion_index]) * 100

    return class_names[emotion_index], confidence, prediction


try:
    model = load_emotion_model()
    class_names = load_class_names()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()


st.title("😊 DeepFER")
st.subheader("Facial Emotion Recognition Using Deep Learning")

st.write(
    "Upload a clear facial image or capture a photo using your camera. "
    "Keep the face centered and close to the camera."
)

input_method = st.radio(
    "Choose input method",
    ["Upload Image", "Use Camera"],
)

input_image = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a facial image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")

else:
    camera_photo = st.camera_input("Capture a facial image")

    if camera_photo is not None:
        input_image = Image.open(camera_photo).convert("RGB")


if input_image is not None:
    st.image(
        input_image,
        caption="Selected Image",
        width="stretch",
    )

    with st.spinner("Detecting emotion..."):
        emotion, confidence, probabilities = predict_emotion(
            input_image,
            model,
            class_names,
        )

    st.success("Emotion detected successfully.")

    st.metric(
        label="Predicted Emotion",
        value=emotion.capitalize(),
        delta=f"{confidence:.2f}% confidence",
    )

    with st.expander("View all emotion probabilities"):
        for name, probability in zip(class_names, probabilities):
            st.write(f"**{name.capitalize()}:** {probability * 100:.2f}%")