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
    WebRtcMode,
    webrtc_streamer,
)

# --------------------------------------------------
# Page setup
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

    video {
        width: 100% !important;
        border-radius: 12px;
    }

    video::-webkit-media-controls,
    video::-webkit-media-controls-enclosure,
    video::-webkit-media-controls-panel {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Paths and labels
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
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
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
            "Haar Cascade face detector could not be loaded."
        )

    return detector


try:
    MODEL = load_model()
    FACE_DETECTOR = load_face_detector()
    CLASS_NAMES = load_class_names()

except Exception as error:
    st.error(f"Unable to start DeepFER: {error}")
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
    equalized = cv2.equalizeHist(gray_image)

    faces = FACE_DETECTOR.detectMultiScale(
        equalized,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(55, 55) if for_live else (30, 30),
    )

    return equalized, faces


def process_uploaded_image(image):
    image_rgb = np.array(image.convert("RGB"))
    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    processed_gray, faces = detect_faces(
        gray,
        for_live=False,
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

    return (
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        results,
    )


# --------------------------------------------------
# Browser live-camera processor
# --------------------------------------------------

class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_box = None
        self.last_emotion = "Detecting..."
        self.last_confidence = 0.0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        self.frame_count += 1

        # Detect the face every second frame.
        should_detect = self.frame_count % 2 == 0

        if should_detect:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

            processed_gray, faces = detect_faces(
                gray,
                for_live=True,
            )

            if len(faces) > 0:
                self.last_box = max(
                    faces,
                    key=lambda box: box[2] * box[3],
                )

                x, y, width, height = self.last_box
                face = processed_gray[
                    y:y + height,
                    x:x + width
                ]

                # Predict every sixth frame to reduce cloud lag.
                if self.frame_count % 6 == 0 and face.size > 0:
                    try:
                        (
                            self.last_emotion,
                            self.last_confidence,
                            _,
                        ) = predict_emotion(face)

                    except Exception:
                        self.last_emotion = "Error"
                        self.last_confidence = 0.0

            else:
                self.last_box = None

        if self.last_box is not None:
            x, y, width, height = self.last_box

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

            cv2.putText(
                image,
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
                image,
                "No face detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("😊 DeepFER")
st.subheader(
    "Facial Emotion Recognition Using Custom CNN"
)

upload_tab, live_tab = st.tabs(
    [
        "📤 Upload Image",
        "📷 Live Camera",
    ]
)

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
            processed_image, results = process_uploaded_image(
                uploaded_image
            )

            if not results:
                st.warning(
                    "No face was detected. Upload a clearer image."
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

                    chart_data = {
                        name.capitalize(): float(score)
                        for name, score in zip(
                            CLASS_NAMES,
                            result["predictions"],
                        )
                    }

                    st.bar_chart(chart_data)


with live_tab:
    st.write(
        "Click **START**, allow camera permission, and face "
        "the camera directly."
    )

    st.info(
        "This uses the visitor's browser camera and works on "
        "Streamlit Cloud."
    )

    rtc_configuration = RTCConfiguration(
        {
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302",
                        "stun:stun1.l.google.com:19302",
                    ]
                }
            ]
        }
    )

    try:
        webrtc_streamer(
            key="deepfer-browser-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=EmotionVideoProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={
                "video": {
                    "width": {
                        "ideal": 320,
                        "max": 480,
                    },
                    "height": {
                        "ideal": 240,
                        "max": 360,
                    },
                    "frameRate": {
                        "ideal": 12,
                        "max": 15,
                    },
                    "facingMode": "user",
                },
                "audio": False,
            },
            video_html_attrs={
                "autoPlay": True,
                "controls": False,
                "muted": True,
                "playsInline": True,
            },
            async_processing=False,
        )

    except Exception as error:
        st.warning(
            "The live camera did not start. Refresh the page, "
            "allow camera permission, and press START again."
        )
        st.caption(f"Camera details: {error}")


st.caption(
    "Supported emotions: Angry, Disgust, Fear, Happy, "
    "Neutral, Sad and Surprise."
)