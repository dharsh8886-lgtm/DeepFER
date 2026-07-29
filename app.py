from pathlib import Path
import threading
import time

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
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 850px;
        padding-top: 2rem;
    }

    img {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Paths and class names
# --------------------------------------------------

MODEL_PATH = Path("models/deepfer_model.keras")
CLASS_NAMES_PATH = Path("models/class_names.txt")

DEFAULT_CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

# --------------------------------------------------
# Load resources
# --------------------------------------------------

def load_class_names():
    if CLASS_NAMES_PATH.exists():
        names = [
            line.strip()
            for line in CLASS_NAMES_PATH.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if names:
            return names

    return DEFAULT_CLASS_NAMES


@st.cache_resource
def load_emotion_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


@st.cache_resource
def load_face_detector():
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(cascade_path)

    if detector.empty():
        raise RuntimeError(
            "The Haar Cascade face detector could not be loaded."
        )

    return detector


try:
    MODEL = load_emotion_model()
    FACE_DETECTOR = load_face_detector()
    CLASS_NAMES = load_class_names()

except Exception as error:
    st.error(f"Unable to start the application: {error}")
    st.stop()


MODEL_LOCK = threading.Lock()

# --------------------------------------------------
# Prediction helpers
# --------------------------------------------------

def predict_emotion(face_gray):
    face_gray = cv2.resize(
        face_gray,
        (48, 48),
        interpolation=cv2.INTER_AREA,
    )

    face_array = face_gray.astype(np.float32) / 255.0
    face_array = np.expand_dims(face_array, axis=-1)
    face_array = np.expand_dims(face_array, axis=0)

    with MODEL_LOCK:
        predictions = MODEL.predict(
            face_array,
            verbose=0,
        )[0]

    emotion_index = int(np.argmax(predictions))
    confidence = float(predictions[emotion_index]) * 100.0

    return (
        CLASS_NAMES[emotion_index],
        confidence,
        predictions,
    )


def detect_faces(gray_image, for_live=False):
    processed_gray = cv2.equalizeHist(gray_image)

    faces = FACE_DETECTOR.detectMultiScale(
        processed_gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(50, 50) if for_live else (30, 30),
    )

    return processed_gray, faces


def draw_predictions(image_bgr, for_live=False):
    gray_image = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    processed_gray, faces = detect_faces(
        gray_image,
        for_live=for_live,
    )

    results = []

    for x, y, width, height in faces:
        face = processed_gray[
            y:y + height,
            x:x + width
        ]

        if face.size == 0:
            continue

        emotion, confidence, predictions = predict_emotion(face)

        results.append(
            {
                "emotion": emotion,
                "confidence": confidence,
                "predictions": predictions,
            }
        )

        label = f"{emotion.capitalize()} {confidence:.1f}%"

        cv2.rectangle(
            image_bgr,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image_bgr,
            label,
            (x, max(y - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if for_live and len(faces) == 0:
        cv2.putText(
            image_bgr,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return image_bgr, results


# --------------------------------------------------
# Interface
# --------------------------------------------------

st.title("😊 DeepFER")
st.subheader(
    "Facial Emotion Recognition Using Custom CNN"
)

upload_tab, live_tab = st.tabs(
    [
        "📤 Upload Image",
        "📷 Local Live Camera",
    ]
)

# --------------------------------------------------
# Upload image
# --------------------------------------------------

with upload_tab:
    st.write(
        "Upload a clear face image to detect the emotion."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file)

        st.image(
            uploaded_image,
            caption="Uploaded image",
            use_container_width=True,
        )

        if st.button(
            "Detect Emotion",
            type="primary",
            key="detect-upload",
        ):
            image_rgb = np.array(
                uploaded_image.convert("RGB")
            )

            image_bgr = cv2.cvtColor(
                image_rgb,
                cv2.COLOR_RGB2BGR,
            )

            processed_bgr, results = draw_predictions(
                image_bgr,
                for_live=False,
            )

            processed_rgb = cv2.cvtColor(
                processed_bgr,
                cv2.COLOR_BGR2RGB,
            )

            if not results:
                st.warning(
                    "No face was detected. Upload a clearer image "
                    "with the full face visible and facing forward."
                )

            else:
                st.image(
                    processed_rgb,
                    caption="Emotion detection result",
                    use_container_width=True,
                )

                for index, result in enumerate(
                    results,
                    start=1,
                ):
                    st.success(
                        f"Face {index}: "
                        f"{result['emotion'].capitalize()} "
                        f"({result['confidence']:.1f}%)"
                    )

                    prediction_values = {
                        class_name.capitalize(): float(score)
                        for class_name, score in zip(
                            CLASS_NAMES,
                            result["predictions"],
                        )
                    }

                    st.bar_chart(prediction_values)


# --------------------------------------------------
# Local OpenCV live camera
# --------------------------------------------------
# --------------------------------------------------
# Local OpenCV live camera
# --------------------------------------------------

with live_tab:
    st.info(
        "This camera mode works on the computer running Streamlit."
    )

    start_camera = st.button(
        "Start Local Camera",
        type="primary",
        key="start-local-camera",
    )

    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    if start_camera:
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        camera.set(cv2.CAP_PROP_FPS, 15)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not camera.isOpened():
            st.error(
                "The webcam could not be opened. Close other apps "
                "using the camera, then try again."
            )

        else:
            status_placeholder.success(
                "Camera started. Refresh the page to stop it."
            )

            frame_count = 0
            last_emotion = "Detecting..."
            last_confidence = 0.0

            try:
                while camera.isOpened():
                    success, frame = camera.read()

                    if not success:
                        st.warning("A camera frame could not be read.")
                        break

                    frame = cv2.flip(frame, 1)

                    gray_image = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2GRAY,
                    )

                    processed_gray, faces = detect_faces(
                        gray_image,
                        for_live=True,
                    )

                    frame_count += 1

                    # Predict less often, but draw the face box on every frame.
                    should_predict = frame_count % 5 == 0

                    if len(faces) > 0:
                        # Use the largest detected face.
                        x, y, width, height = max(
                            faces,
                            key=lambda box: box[2] * box[3],
                        )

                        face = processed_gray[
                            y:y + height,
                            x:x + width
                        ]

                        if should_predict and face.size > 0:
                            try:
                                (
                                    last_emotion,
                                    last_confidence,
                                    _,
                                ) = predict_emotion(face)

                            except Exception:
                                last_emotion = "Error"
                                last_confidence = 0.0

                        label = (
                            f"{last_emotion.capitalize()} "
                            f"{last_confidence:.1f}%"
                        )

                        cv2.rectangle(
                            frame,
                            (x, y),
                            (x + width, y + height),
                            (0, 255, 0),
                            2,
                        )

                        cv2.putText(
                            frame,
                            label,
                            (x, max(y - 10, 30)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )

                    else:
                        cv2.putText(
                            frame,
                            "No face detected",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    frame_rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB,
                    )

                    frame_placeholder.image(
                        frame_rgb,
                        channels="RGB",
                        use_container_width=True,
                    )

                    time.sleep(0.02)

            finally:
                camera.release()
                status_placeholder.info("Camera stopped.")


st.caption(
    "Supported emotions: Angry, Disgust, Fear, Happy, "
    "Neutral, Sad and Surprise."
)