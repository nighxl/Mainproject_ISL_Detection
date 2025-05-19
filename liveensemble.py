import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib
import cv2
import matplotlib.pyplot as plt

# ========== Configure TensorFlow for GPU ==========
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU detected and configured.")
    except RuntimeError as e:
        print("Error configuring GPU:", e)
else:
    print("No GPU detected, using CPU.")

# Enable mixed precision training to reduce memory usage
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

# Path to the trained SVM model
MODEL_PATH = r"C:\Users\rajni\OneDrive\Desktop\Phase2 codess\modeling/ISL_SVM_Modelnew.pkl"

# Path to the scaler
SCALER_PATH = r"C:\Users\rajni\OneDrive\Desktop\Phase2 codess\modeling/scalernew.pkl"

# Input sizes for ResNet50 and InceptionV3
IMG_SIZE_RESNET = (224, 224)  # Input size for ResNet50
IMG_SIZE_INCEPTION = (299, 299)  # Input size for InceptionV3

# Load the trained SVM model
print("Loading the trained SVM model...")
svm = joblib.load(MODEL_PATH)
print("SVM model loaded.")

# Load the scaler
print("Loading the scaler...")
scaler = joblib.load(SCALER_PATH)
print("Scaler loaded.")

# Load ResNet50 for feature extraction
with tf.device('/CPU:0'):  # Offload to CPU if GPU memory is insufficient
    base_model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Load InceptionV3 for feature extraction
with tf.device('/CPU:0'):  # Offload to CPU if GPU memory is insufficient
    base_model_inception = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

# Function to preprocess and predict an image
def predict_image(image, base_model_resnet, base_model_inception, svm, scaler, img_size_resnet, img_size_inception):
    # Resize and preprocess the image for ResNet50
    image_resnet = cv2.resize(image, img_size_resnet)
    image_resnet = img_to_array(image_resnet) / 255.0
    image_resnet = np.expand_dims(image_resnet, axis=0)  # Add batch dimension

    # Resize and preprocess the image for InceptionV3
    image_inception = cv2.resize(image, img_size_inception)
    image_inception = img_to_array(image_inception) / 255.0
    image_inception = np.expand_dims(image_inception, axis=0)  # Add batch dimension

    # Extract features
    features_resnet = base_model_resnet.predict(image_resnet)
    features_inception = base_model_inception.predict(image_inception)

    # Combine features
    features_combined = np.concatenate((features_resnet, features_inception), axis=1)

    # Normalize features
    features_normalized = scaler.transform(features_combined)

    # Make a prediction
    pred_label = svm.predict(features_normalized)[0]
    return pred_label

# Map the predicted label to the class name
label_map = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I",
    19: "J", 20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R",
    28: "S", 29: "T", 30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z"
}

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting live detection. Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Predict the sign in the frame
    pred_label = predict_image(frame, base_model_resnet, base_model_inception, svm, scaler, IMG_SIZE_RESNET, IMG_SIZE_INCEPTION)
    pred_class = label_map[pred_label]

    # Display the predicted class on the frame
    cv2.putText(frame, f"Predicted: {pred_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Live Sign Language Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()