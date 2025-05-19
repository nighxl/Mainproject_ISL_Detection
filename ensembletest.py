import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib
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
MODEL_PATH = r"C:\Users\rajni\OneDrive\Desktop\Phase2 codess\modeling/ISL_SVM_Model.pkl"

# Path to the scaler
SCALER_PATH = r"C:\Users\rajni\OneDrive\Desktop\Phase2 codess\modeling/scaler.pkl"

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
def predict_image(image_path, base_model_resnet, base_model_inception, svm, scaler, img_size_resnet, img_size_inception):
    # Load and preprocess the image for ResNet50
    image_resnet = load_img(image_path, target_size=img_size_resnet)
    image_resnet = img_to_array(image_resnet) / 255.0
    image_resnet = np.expand_dims(image_resnet, axis=0)  # Add batch dimension

    # Load and preprocess the image for InceptionV3
    image_inception = load_img(image_path, target_size=img_size_inception)
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

# Example usage
image_path = r"C:\Users\rajni\OneDrive\Desktop\Phase2 codess\1009.jpg"  # Replace with the path to your image
prediction = predict_image(image_path, base_model_resnet, base_model_inception, svm, scaler, IMG_SIZE_RESNET, IMG_SIZE_INCEPTION)

# Map the predicted label to the class name
label_map = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I",
    19: "J", 20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R",
    28: "S", 29: "T", 30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z"
}

pred_class = label_map[prediction]
print(f"Predicted: {pred_class}")

# Display the image
image = load_img(image_path, target_size=IMG_SIZE_RESNET)
plt.imshow(image)
plt.title(f"Predicted: {pred_class}")
plt.axis('off')
plt.show()