import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# -----------------------
# Paths
# -----------------------
MODEL_PATH = "models/deepfer_model.keras"
TEST_DIR = "dataset/test"
IMG_SIZE = (48, 48)
BATCH_SIZE = 32

# -----------------------
# Load Model
# -----------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------
# Load Test Dataset
# -----------------------
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="categorical"
)

# Normalize images
test_ds = test_ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

class_names = test_ds.class_names

# -----------------------
# Evaluate
# -----------------------
loss, accuracy = model.evaluate(test_ds)
print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")

# -----------------------
# Predictions
# -----------------------
y_true = np.concatenate([y.numpy() for x, y in test_ds])
y_true = np.argmax(y_true, axis=1)

y_pred = model.predict(test_ds)
y_pred = np.argmax(y_pred, axis=1)

# -----------------------
# Classification Report
# -----------------------
print("\nClassification Report\n")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)

# -----------------------
# Confusion Matrix
# -----------------------
os.makedirs("graphs", exist_ok=True)

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

plt.figure(figsize=(8,8))
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("graphs/confusion_matrix.png")
plt.show()