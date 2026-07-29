from pathlib import Path
import threading

import cv2
import gradio as gr
import numpy as np
import tensorflow as tf

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
            for line in CLASS_NAMES_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if names:
            return names
    return DEFAULT_CLASS_NAMES

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")

MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_DETECTOR = cv2.CascadeClassifier(cascade_path)

if FACE_DETECTOR.empty():
    raise RuntimeError("The Haar Cascade face detector could not be loaded.")

CLASS_NAMES = load_class_names()
MODEL_LOCK = threading.Lock()

def predict_emotion(face_gray):
    face_gray = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
    face_array = face_gray.astype(np.float32) / 255.0
    face_array = np.expand_dims(face_array, axis=-1)
    face_array = np.expand_dims(face_array, axis=0)

    with MODEL_LOCK:
        predictions = MODEL.predict(face_array, verbose=0)[0]

    emotion_index = int(np.argmax(predictions))
    return (
        CLASS_NAMES[emotion_index],
        float(predictions[emotion_index]),
        predictions,
    )

def process_image(image):
    if image is None:
        return None, {}

    image_rgb = image.astype(np.uint8)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)

    faces = FACE_DETECTOR.detectMultiScale(
        equalized,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(40, 40),
    )

    if len(faces) == 0:
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
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), {}

    x, y, width, height = max(
        faces,
        key=lambda box: box[2] * box[3],
    )

    face = equalized[y:y + height, x:x + width]
    emotion, confidence, predictions = predict_emotion(face)

    label = f"{emotion.capitalize()} {confidence * 100:.1f}%"

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

    scores = {
        name.capitalize(): float(score)
        for name, score in zip(CLASS_NAMES, predictions)
    }

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), scores

with gr.Blocks(title="DeepFER") as demo:
    gr.Markdown(
        """
        # 😊 DeepFER
        ## Facial Emotion Recognition Using Custom CNN
        Use the webcam or upload an image to detect facial emotion.
        """
    )

    with gr.Tab("📷 Live Camera"):
        live_input = gr.Image(
            sources=["webcam"],
            type="numpy",
            label="Webcam",
            streaming=True,
        )
        live_output = gr.Image(
            type="numpy",
            label="Live emotion detection",
        )
        live_scores = gr.Label(
            num_top_classes=7,
            label="Emotion probabilities",
        )

        live_input.stream(
            fn=process_image,
            inputs=live_input,
            outputs=[live_output, live_scores],
            time_limit=120,
            stream_every=0.5,
        )

    with gr.Tab("📤 Upload Image"):
        upload_input = gr.Image(
            sources=["upload"],
            type="numpy",
            label="Upload image",
        )
        upload_button = gr.Button("Detect Emotion", variant="primary")
        upload_output = gr.Image(
            type="numpy",
            label="Detection result",
        )
        upload_scores = gr.Label(
            num_top_classes=7,
            label="Emotion probabilities",
        )

        upload_button.click(
            fn=process_image,
            inputs=upload_input,
            outputs=[upload_output, upload_scores],
        )

    gr.Markdown(
        """
        Supported emotions: Angry, Disgust, Fear, Happy,
        Neutral, Sad and Surprise.
        """
    )

if __name__ == "__main__":
    demo.launch()