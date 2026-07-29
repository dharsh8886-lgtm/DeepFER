# DeepFER - Facial Emotion Recognition Using CNN

## Overview
DeepFER is a facial emotion recognition system developed using a Convolutional Neural Network (CNN). It detects human facial expressions from live webcam input or uploaded images and classifies them into one of seven emotions.

## Features
- Live webcam emotion detection
- Image upload emotion detection
- Face detection using OpenCV
- CNN-based emotion classification
- User-friendly Gradio web interface
- Deployable on Render

## Emotion Classes
- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Model Performance
- Test Accuracy: **55.88%**

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- Gradio
- NumPy

## Project Structure
```
DeepFER/
│── app.py
│── train.py
│── requirements.txt
│── models/
│   ├── deepfer_model.keras
│   └── class_names.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python app.py
```

To retrain the model:

```bash
python train.py
```

## Author

Baby Dharshini S
