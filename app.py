from pathlib import Path
import threading

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    webrtc_streamer,
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="DeepFER",
    page_icon="😊",
    layout="centered",
)

# --------------------------------------------------
# File paths and class names
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
# Load class names
# --------------------------------------------------

def load_class_names():
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
            names = [
                line.strip()
                for line in file
                if line.strip()
            ]

        if names:
            return names

    return DEFAULT_CLASS_NAMES


# --------------------------------------------------
# Load CNN model
# --------------------------------------------------

@st.cache_resource
def load_emotion_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# Load OpenCV face detector
# --------------------------------------------------

@st.cache_resource
def load_face_detector():
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError(
            "OpenCV was not installed correctly. "
            "Check requirements.txt and reboot the app."
        )

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


# --------------------------------------------------
# Load project resources
# --------------------------------------------------

try:
    MODEL = load_emotion_model()
    FACE_DETECTOR = load_face_detector()
    CLASS_NAMES = load_class_names()

except Exception as error:
    st.error(f"Unable to start the application: {error}")
    st.stop()


MODEL_LOCK = threading.Lock()

# --------------------------------------------------
# Emotion prediction function
# --------------------------------------------------

def predict_emotion(face_gray):
    face_gray = cv2.resize(face_gray, (48, 48))

    face_array = face_gray.astype(np.float32) / 255.0

    # Final shape: (1, 48, 48, 1)
    face_array = np.expand_dims(face_array, axis=-1)
    face_array = np.expand_dims(face_array, axis=0)

    with MODEL_LOCK:
        predictions = MODEL.predict(
            face_array,
            verbose=0,
        )[0]

    emotion_index = int(np.argmax(predictions))
    confidence = float(predictions[emotion_index]) * 100

    return (
        CLASS_NAMES[emotion_index],
        confidence,
        predictions,
    )


# --------------------------------------------------
# Detect emotions in an uploaded image
# --------------------------------------------------

def process_uploaded_image(image):
    image_rgb = np.array(image.convert("RGB"))

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    gray_image = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    faces = FACE_DETECTOR.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    results = []

    for x, y, width, height in faces:
        face = gray_image[
            y:y + height,
            x:x + width
        ]

        if face.size == 0:
            continue

        emotion, confidence, predictions = predict_emotion(
            face
        )

        results.append(
            {
                "emotion": emotion,
                "confidence": confidence,
                "predictions": predictions,
            }
        )

        label = (
            f"{emotion.capitalize()} "
            f"{confidence:.1f}%"
        )

        cv2.rectangle(
            image_bgr,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2,
        )

        label_y = max(y - 10, 30)

        cv2.putText(
            image_bgr,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    processed_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return processed_rgb, results


# --------------------------------------------------
# Live video processor
# --------------------------------------------------

class EmotionVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.frame_count = 0
        self.last_emotion = "Detecting..."
        self.last_confidence = 0.0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Mirror webcam image
        image = cv2.flip(image, 1)

        gray_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        faces = FACE_DETECTOR.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        self.frame_count += 1

        # Predict every second frame
        should_predict = self.frame_count % 2 == 0

        for x, y, width, height in faces:
            face = gray_image[
                y:y + height,
                x:x + width
            ]

            if should_predict and face.size > 0:
                try:
                    emotion, confidence, _ = predict_emotion(
                        face
                    )

                    self.last_emotion = emotion
                    self.last_confidence = confidence

                except Exception:
                    self.last_emotion = "Error"
                    self.last_confidence = 0.0

            label = (
                f"{self.last_emotion.capitalize()} "
                f"{self.last_confidence:.1f}%"
            )

            cv2.rectangle(
                image,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2,
            )

            label_y = max(y - 10, 30)

            cv2.putText(
                image,
                label,
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if len(faces) == 0:
            cv2.putText(
                image,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )


# --------------------------------------------------
# Streamlit interface
# --------------------------------------------------

st.title("😊 DeepFER")

st.subheader(
    "Facial Emotion Recognition Using Custom CNN"
)

upload_tab, live_tab = st.tabs(
    [
        "📤 Upload Image",
        "📷 Live Detection",
    ]
)

# --------------------------------------------------
# Upload image tab
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
        ):
            processed_image, results = (
                process_uploaded_image(uploaded_image)
            )

            if not results:
                st.warning(
                    "No face was detected. Upload a clearer "
                    "image with the face visible."
                )

            else:
                st.image(
                    processed_image,
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
# Live detection tab
# --------------------------------------------------

with live_tab:
    st.write(
        "Click **START**, allow camera permission and look "
        "directly at the camera."
    )

    st.info(
        "Use good lighting and keep your face close to the "
        "camera for better predictions."
    )

    RTC_CONFIGURATION = RTCConfiguration(
        {
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302"
                    ]
                }
            ]
        }
    )

    webrtc_streamer(
        key="deepfer-live-camera",
        video_processor_factory=EmotionVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

st.caption(
    "Supported emotions: Angry, Disgust, Fear, Happy, "
    "Neutral, Sad and Surprise."
)