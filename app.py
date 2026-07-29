from pathlib import Path
import os
import threading

import cv2
import gradio as gr
import numpy as np
import tensorflow as tf

# --------------------------------------------------
# Configuration
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

CONFIDENCE_THRESHOLD = 0.40
SMOOTHING_WINDOW = 5
MAX_FACES = 3

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


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file was not found: {MODEL_PATH}"
    )

MODEL = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

cascade_path = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

FACE_DETECTOR = cv2.CascadeClassifier(cascade_path)

if FACE_DETECTOR.empty():
    raise RuntimeError(
        "The Haar Cascade face detector could not be loaded."
    )

CLASS_NAMES = load_class_names()
MODEL_LOCK = threading.Lock()

# --------------------------------------------------
# Model inference
# --------------------------------------------------

def predict_emotion_batch(face_grays):
    batch = []

    for face_gray in face_grays:
        resized = cv2.resize(
            face_gray,
            (48, 48),
            interpolation=cv2.INTER_AREA,
        )

        array = resized.astype(np.float32) / 255.0
        array = np.expand_dims(array, axis=-1)
        batch.append(array)

    batch_array = np.stack(batch, axis=0)

    with MODEL_LOCK:
        predictions = MODEL.predict(
            batch_array,
            verbose=0,
        )

    return predictions


def smooth_probabilities(history, new_probs):
    history = list(history) if history else []
    history.append(new_probs.tolist())
    history = history[-SMOOTHING_WINDOW:]

    averaged = np.mean(
        np.asarray(history),
        axis=0,
    )

    return averaged, history


def draw_label(image, x, y, text, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.72
    thickness = 2

    (text_width, text_height), _ = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )

    top = max(y - text_height - 18, 0)
    bottom = max(y, text_height + 18)

    cv2.rectangle(
        image,
        (x, top),
        (x + text_width + 14, bottom),
        color,
        -1,
    )

    cv2.putText(
        image,
        text,
        (x + 7, bottom - 7),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def process_frame(image, history_state):
    if image is None:
        return None, {}, history_state

    image_rgb = np.asarray(image).astype(np.uint8)

    if image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
        image_rgb = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGBA2RGB,
        )

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    equalized = cv2.equalizeHist(gray)

    faces = FACE_DETECTOR.detectMultiScale(
        equalized,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(55, 55),
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

        processed = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        return processed, {}, history_state

    faces = sorted(
        faces,
        key=lambda box: box[2] * box[3],
        reverse=True,
    )[:MAX_FACES]

    face_crops = [
        equalized[y:y + h, x:x + w]
        for x, y, w, h in faces
    ]

    predictions = predict_emotion_batch(face_crops)

    smoothed_probs, history_state = smooth_probabilities(
        history_state,
        predictions[0],
    )

    for index, (x, y, w, h) in enumerate(faces):
        probs = (
            smoothed_probs
            if index == 0
            else predictions[index]
        )

        emotion_index = int(np.argmax(probs))
        confidence = float(probs[emotion_index])
        emotion = CLASS_NAMES[emotion_index]

        display_name = (
            "Uncertain"
            if confidence < CONFIDENCE_THRESHOLD
            else emotion.capitalize()
        )

        label = f"{display_name} {confidence * 100:.0f}%"
        color = (0, 255, 0)

        cv2.rectangle(
            image_bgr,
            (x, y),
            (x + w, y + h),
            color,
            3,
        )

        draw_label(
            image_bgr,
            x,
            y,
            label,
            color,
        )

    scores = {
        name.capitalize(): float(score)
        for name, score in zip(
            CLASS_NAMES,
            smoothed_probs,
        )
    }

    processed = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    return processed, scores, history_state


def reset_history():
    return []


# --------------------------------------------------
# UI styling
# --------------------------------------------------

CUSTOM_CSS = """
:root {
    --deepfer-orange: #ff6b00;
    --deepfer-green: #16a34a;
    --deepfer-red: #dc2626;
}

.gradio-container {
    max-width: 1180px !important;
    margin: auto !important;
    padding-top: 22px !important;
}

#app-header {
    margin-bottom: 10px;
}

#camera-shell {
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 14px;
    background: white;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

/* Keep webcam mounted for browser capture, but visually hide it */
#hidden-webcam {
    position: absolute !important;
    left: -10000px !important;
    top: -10000px !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

#processed-camera {
    max-width: 900px;
    margin: 0 auto;
}

#processed-camera img {
    object-fit: contain !important;
    max-height: 560px !important;
    border-radius: 12px !important;
}

#camera-controls {
    justify-content: center !important;
    gap: 14px !important;
    margin-top: 12px !important;
}

#start-camera button {
    background: var(--deepfer-green) !important;
    color: white !important;
    min-width: 150px !important;
    border: none !important;
}

#stop-camera button {
    background: var(--deepfer-red) !important;
    color: white !important;
    min-width: 150px !important;
    border: none !important;
}

#probability-panel {
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 10px;
    margin-top: 14px;
}

footer {
    display: none !important;
}
"""

START_CAMERA_JS = """
() => {
    const root = document.querySelector('#hidden-webcam');
    if (!root) return [];

    const buttons = Array.from(root.querySelectorAll('button'));

    const target = buttons.find((button) => {
        const text = (
            (button.innerText || '') + ' ' +
            (button.getAttribute('aria-label') || '') + ' ' +
            (button.title || '')
        ).toLowerCase();

        return (
            text.includes('access webcam') ||
            text.includes('webcam') ||
            text.includes('record') ||
            text.includes('start')
        );
    });

    if (target) {
        target.click();
    }

    return [];
}
"""

STOP_CAMERA_JS = """
() => {
    const root = document.querySelector('#hidden-webcam');
    if (!root) return [];

    const buttons = Array.from(root.querySelectorAll('button'));

    const target = buttons.find((button) => {
        const text = (
            (button.innerText || '') + ' ' +
            (button.getAttribute('aria-label') || '') + ' ' +
            (button.title || '')
        ).toLowerCase();

        return text.includes('stop');
    });

    if (target) {
        target.click();
    }

    return [];
}
"""

# --------------------------------------------------
# Gradio app
# --------------------------------------------------

with gr.Blocks(
    title="DeepFER",
    css=CUSTOM_CSS,
) as demo:
    gr.Markdown(
        """
        # 😊 DeepFER
        ## Facial Emotion Recognition Using Custom CNN
        Use the webcam or upload an image to detect facial emotion.
        """,
        elem_id="app-header",
    )

    history_state = gr.State([])

    with gr.Tab("📷 Live Camera"):
        # Hidden source component that captures browser webcam frames.
        hidden_webcam = gr.Image(
            sources=["webcam"],
            type="numpy",
            label="Webcam Source",
            streaming=True,
            elem_id="hidden-webcam",
        )

        with gr.Column(elem_id="camera-shell"):
            processed_output = gr.Image(
                type="numpy",
                label="Live Emotion Detection",
                height=560,
                interactive=False,
                elem_id="processed-camera",
            )

            with gr.Row(elem_id="camera-controls"):
                start_button = gr.Button(
                    "▶ Start",
                    variant="primary",
                    elem_id="start-camera",
                )

                stop_button = gr.Button(
                    "■ Stop",
                    variant="stop",
                    elem_id="stop-camera",
                )

        live_scores = gr.Label(
            num_top_classes=7,
            label="Emotion Probabilities (smoothed)",
            elem_id="probability-panel",
        )

        hidden_webcam.stream(
            fn=process_frame,
            inputs=[
                hidden_webcam,
                history_state,
            ],
            outputs=[
                processed_output,
                live_scores,
                history_state,
            ],
            stream_every=0.1,
            time_limit=None,
            concurrency_limit=1,
        )

        start_button.click(
            fn=reset_history,
            inputs=[],
            outputs=[history_state],
            js=START_CAMERA_JS,
        )

        stop_button.click(
            fn=reset_history,
            inputs=[],
            outputs=[history_state],
            js=STOP_CAMERA_JS,
        )

    with gr.Tab("📤 Upload Image"):
        with gr.Row():
            upload_input = gr.Image(
                sources=["upload"],
                type="numpy",
                label="Upload Image",
                height=420,
            )

            upload_output = gr.Image(
                type="numpy",
                label="Detection Result",
                height=420,
                interactive=False,
            )

        upload_button = gr.Button(
            "Detect Emotion",
            variant="primary",
        )

        upload_scores = gr.Label(
            num_top_classes=7,
            label="Emotion Probabilities",
        )

        upload_button.click(
            fn=process_frame,
            inputs=[
                upload_input,
                history_state,
            ],
            outputs=[
                upload_output,
                upload_scores,
                history_state,
            ],
            concurrency_limit=1,
        )

    gr.Markdown(
        """
        💡 **Supported emotions:** Angry, Disgust, Fear,
        Happy, Neutral, Sad and Surprise.
        """
    )


if __name__ == "__main__":
    demo.queue(
        max_size=10,
        default_concurrency_limit=1,
    ).launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get("PORT", 7860)
        ),
    )