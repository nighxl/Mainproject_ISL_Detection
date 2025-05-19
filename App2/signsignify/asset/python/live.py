import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, InceptionV3
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.preprocessing.image import img_to_array
import joblib
import cv2

# ========== Configure TensorFlow for GPU ==========
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU detected and configured.")
    except RuntimeError as e:
        print("⚠️ Error configuring GPU:", e)
else:
    print("⚠️ No GPU detected, using CPU.")

# ========== Load Model and Scaler ==========
MODEL_PATH = "ISL_SVM_Model.pkl"
SCALER_PATH = "scaler.pkl"

print("🔄 Loading the trained SVM model...")
svm = joblib.load(MODEL_PATH)
print("✅ SVM model loaded.")

print("🔄 Loading the scaler...")
scaler = joblib.load(SCALER_PATH)
print("✅ Scaler loaded.")

# Load ResNet50 and InceptionV3 for feature extraction
with tf.device('/CPU:0'):  
    base_model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    base_model_inception = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

# ========== Prediction Function ==========
def predict_image(image):
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize and preprocess for ResNet50
    image_resnet = cv2.resize(image, (224, 224))
    image_resnet = img_to_array(image_resnet)
    image_resnet = np.expand_dims(image_resnet, axis=0)
    image_resnet = resnet_preprocess(image_resnet)

    # Resize and preprocess for InceptionV3
    image_inception = cv2.resize(image, (299, 299))
    image_inception = img_to_array(image_inception)
    image_inception = np.expand_dims(image_inception, axis=0)
    image_inception = inception_preprocess(image_inception)

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

# ========== Label Mapping ==========
label_map = {i: chr(48 + i) if i < 10 else chr(65 + i - 10) for i in range(36)}

# ========== Initialize Webcam ==========
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam. Check camera permissions.")
    exit()

print("🎥 Starting live detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)  # Flip for a mirror effect

    # Predict the sign in the frame
    pred_label = predict_image(frame)
    pred_class = label_map[pred_label]

    # Display the predicted class on the frame
    cv2.putText(frame, f"Predicted: {pred_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Live Sign Language Detection", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
