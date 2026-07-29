import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

st.title("DeepFER Test")

st.write("TensorFlow:", tf.__version__)
st.write("OpenCV:", cv2.__version__)
st.write("NumPy:", np.__version__)
st.success("App started successfully!")
