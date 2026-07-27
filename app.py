from pathlib import Path

import cv2
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
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_face_detector():
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError("Face detector could not be loaded.")

    return detector


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


def detect_largest_face(image_rgb, detector):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return None, None

    # Select the largest detected face
    x, y, width, height = max(
        faces,
        key=lambda face: face[2] * face[3],
    )

    # Add a small margin around the face
    margin_x = int(width * 0.12)
    margin_y = int(height * 0.12)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(image_rgb.shape[1], x + width + margin_x)
    y2 = min(image_rgb.shape[0], y + height + margin_y)

    face_crop = gray[y1:y2, x1:x2]

    return face_crop, (x1, y1, x2, y2)


def prepare_face(face_crop):
    face_crop = cv2.resize(face_crop, (48, 48))

    face_array = face_crop.astype("float32") / 255.0

    # Shape: (1, 48, 48, 1)
    face_array = np.expand_dims(face_array, axis=-1)
    face_array = np.expand_dims(face_array, axis=0)

    return face_array


def predict_emotion(face_crop, model, class_names):
    input_array = prepare_face(face_crop)

    predictions = model.predict(input_array, verbose=0)[0]

    emotion_index = int(np.argmax(predictions))
    confidence = float(predictions[emotion_index]) * 100

    return (
        class_names[emotion_index],
        confidence,
        predictions,
    )


try:
    model = load_model()
    face_detector = load_face_detector()
    class_names = load_class_names()
except Exception as error:
    st.error(f"Unable to start the application: {error}")
    st.stop()


st.title("😊 DeepFER")
st.subheader("Facial Emotion Recognition Using Deep Learning")

st.write(
    "Upload a facial image or capture a photo. "
    "The application will detect your face before predicting the emotion."
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
        input_image = Image.open(uploaded_file)

else:
    camera_photo = st.camera_input("Capture a facial image")

    if camera_photo is not None:
        input_image = Image.open(camera_photo)


if input_image is not None:
    input_image = ImageOps.exif_transpose(input_image).convert("RGB")
    image_array = np.array(input_image)

    st.image(
        input_image,
        caption="Selected Image",
        use_container_width=True,
    )

    with st.spinner("Detecting face and emotion..."):
        face_crop, face_box = detect_largest_face(
            image_array,
            face_detector,
        )

    if face_crop is None:
        st.warning(
            "No face was detected. Use a clear, front-facing photo "
            "with good lighting."
        )

    else:
        emotion, confidence, probabilities = predict_emotion(
            face_crop,
            model,
            class_names,
        )

        x1, y1, x2, y2 = face_box

        output_image = image_array.copy()

        cv2.rectangle(
            output_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
        )

        label = f"{emotion.capitalize()}: {confidence:.1f}%"

        cv2.putText(
            output_image,
            label,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        st.image(
            output_image,
            caption="Detected Face and Emotion",
            use_container_width=True,
        )

        st.success("Face and emotion detected successfully.")

        st.metric(
            label="Predicted Emotion",
            value=emotion.capitalize(),
            delta=f"{confidence:.2f}% confidence",
        )

        with st.expander("View all emotion probabilities"):
            for name, probability in zip(
                class_names,
                probabilities,
            ):
                st.write(
                    f"**{name.capitalize()}:** "
                    f"{probability * 100:.2f}%"
                )