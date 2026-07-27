import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="DeepFER",
    page_icon="😊",
    layout="centered"
)

# --------------------------------------------------
# Constants
# --------------------------------------------------

MODEL_PATH = "models/deepfer_model.keras"

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_emotion_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_emotion_model()
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()

# --------------------------------------------------
# Load face detector
# --------------------------------------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# --------------------------------------------------
# Emotion prediction function
# --------------------------------------------------

def predict_emotion(image: np.ndarray):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50)
    )

    results = []

    for x, y, width, height in faces:
        face = gray_image[y:y + height, x:x + width]

        face = cv2.resize(face, (48, 48))

        # Normalize exactly as during model training
        face = face.astype("float32") / 255.0

        # Shape becomes: (1, 48, 48, 1)
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        predictions = model.predict(face, verbose=0)[0]

        emotion_index = int(np.argmax(predictions))
        confidence = float(predictions[emotion_index]) * 100

        results.append({
            "emotion": CLASS_NAMES[emotion_index],
            "confidence": confidence,
            "box": (x, y, width, height)
        })

    return results

# --------------------------------------------------
# Draw detected faces
# --------------------------------------------------

def draw_results(image: np.ndarray, results):
    output_image = image.copy()

    for result in results:
        x, y, width, height = result["box"]
        emotion = result["emotion"].capitalize()
        confidence = result["confidence"]

        label = f"{emotion}: {confidence:.1f}%"

        cv2.rectangle(
            output_image,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output_image,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return output_image

# --------------------------------------------------
# User interface
# --------------------------------------------------

st.title("😊 DeepFER")
st.subheader("Facial Emotion Recognition Using Deep Learning")

st.write(
    "Upload a facial image or capture a photo using your camera. "
    "The CNN model will predict the detected emotion."
)

input_method = st.radio(
    "Choose input method",
    ["Upload Image", "Use Camera"]
)

input_image = None

if input_method == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload a facial image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")

else:
    camera_photo = st.camera_input("Capture a facial image")

    if camera_photo is not None:
        input_image = Image.open(camera_photo).convert("RGB")

# --------------------------------------------------
# Run prediction
# --------------------------------------------------

if input_image is not None:
    image_array = np.array(input_image)

    st.image(
        input_image,
        caption="Selected Image",
        use_container_width=True
    )

    with st.spinner("Detecting emotion..."):
        results = predict_emotion(image_array)

    if not results:
        st.warning(
            "No face was detected. Try using a clear, front-facing image "
            "with good lighting."
        )
    else:
        output_image = draw_results(image_array, results)

        st.image(
            output_image,
            caption="Emotion Detection Result",
            use_container_width=True
        )

        st.success("Emotion detected successfully.")

        for index, result in enumerate(results, start=1):
            st.write(
                f"**Face {index}:** "
                f"{result['emotion'].capitalize()} — "
                f"{result['confidence']:.2f}% confidence"
            )