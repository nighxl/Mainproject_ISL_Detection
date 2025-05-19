import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'  # Enable async memory allocator

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import time
import gc
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

# Path to dataset
DATASET_PATH = r"C:\Users\rajni\OneDrive\Desktop\Phase2 codess\filtered_images"
IMG_SIZE_RESNET = (224, 224)  # Input size for ResNet50
IMG_SIZE_INCEPTION = (299, 299)  # Input size for InceptionV3
BATCH_SIZE = 16  # Batch size for processing

# ========== Phase 2: Load Data, Extract Features, Train SVM, Save Model ==========

# Function to load and preprocess an image for ResNet50
def load_and_preprocess_image_resnet(img_path, label):
    image = load_img(img_path, target_size=IMG_SIZE_RESNET)
    image = img_to_array(image) / 255.0
    return image, label

# Function to load and preprocess an image for InceptionV3
def load_and_preprocess_image_inception(img_path, label):
    image = load_img(img_path, target_size=IMG_SIZE_INCEPTION)
    image = img_to_array(image) / 255.0
    return image, label

# Create a tf.data.Dataset for efficient data loading and preprocessing
def create_dataset(dataset_path, img_size, batch_size, preprocess_fn):
    image_paths = []
    labels = []
    classes = sorted(os.listdir(dataset_path))
    label_map = {cls: i for i, cls in enumerate(classes)}

    # Collect all image paths and labels
    for cls in classes:
        class_path = os.path.join(dataset_path, cls)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image_paths.append(img_path)
            labels.append(label_map[cls])

    # Create a tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda img_path, label: tf.numpy_function(
        preprocess_fn, [img_path, label], (tf.float32, tf.int32)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset, label_map, len(image_paths)

# Load dataset for ResNet50
print("Loading dataset for ResNet50...")
dataset_resnet, label_map, total_images = create_dataset(DATASET_PATH, IMG_SIZE_RESNET, BATCH_SIZE, load_and_preprocess_image_resnet)
print(f"Total images to process: {total_images}")

# Load dataset for InceptionV3
print("Loading dataset for InceptionV3...")
dataset_inception, _, _ = create_dataset(DATASET_PATH, IMG_SIZE_INCEPTION, BATCH_SIZE, load_and_preprocess_image_inception)

# Load ResNet50 for feature extraction
with tf.device('/CPU:0'):  # Offload to CPU if GPU memory is insufficient
    base_model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Load InceptionV3 for feature extraction
with tf.device('/CPU:0'):  # Offload to CPU if GPU memory is insufficient
    base_model_inception = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

# Function to extract features and save them to disk
def extract_features_and_save(dataset, base_model, total_images, model_name):
    FEATURES = []
    LABELS = []
    processed_images = 0
    start_time = time.time()

    for batch in dataset:
        images, labels = batch
        features = base_model.predict(images)
        FEATURES.extend(features)
        LABELS.extend(labels.numpy())
        processed_images += len(images)

        # Clear GPU memory
        del images, labels, features
        gc.collect()

        # Progress monitoring
        elapsed_time = time.time() - start_time
        estimated_time = (elapsed_time / processed_images) * (total_images - processed_images)
        print(f"{model_name}: Processed {processed_images}/{total_images} images. Estimated time remaining: {estimated_time:.2f} seconds")

    # Save features and labels to disk
    np.save(f"features_{model_name}.npy", np.array(FEATURES))
    np.save(f"labels_{model_name}.npy", np.array(LABELS))
    print(f"{model_name}: Features and labels saved to disk.")

# Extract features using ResNet50
print("Extracting features using ResNet50...")
extract_features_and_save(dataset_resnet, base_model_resnet, total_images, "resnet")

# Extract features using InceptionV3
print("Extracting features using InceptionV3...")
extract_features_and_save(dataset_inception, base_model_inception, total_images, "inception")

# Load features and labels from disk
FEATURES_RESNET = np.load("features_resnet.npy")
FEATURES_INCEPTION = np.load("features_inception.npy")
LABELS = np.load("labels_resnet.npy")  # Labels are the same for both models

# Combine features from both models
FEATURES_COMBINED = np.concatenate((FEATURES_RESNET, FEATURES_INCEPTION), axis=1)

# Normalize features
scaler = StandardScaler()
FEATURES_NORMALIZED = scaler.fit_transform(FEATURES_COMBINED)

# Save the scaler
SCALER_PATH = r"C:\Users\rajni\OneDrive\Desktop\Phase2 codess\modeling/scaler.pkl"
joblib.dump(scaler, SCALER_PATH)
print(f"Scaler saved to {SCALER_PATH}")

# Train SVM classifier
print("Training SVM model...")
svm = SVC(kernel='linear', probability=True)
svm.fit(FEATURES_NORMALIZED, LABELS)
print("SVM model training complete.")

# Save the trained SVM model
MODEL_PATH = r"C:\Users\rajni\OneDrive\Desktop\Phase2 codess\modeling/ISL_SVM_Model.pkl"
joblib.dump(svm, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# ========== Phase 3: Evaluate Model ==========
PRED_LABELS = svm.predict(FEATURES_NORMALIZED)
print("Accuracy:", accuracy_score(LABELS, PRED_LABELS))
print("Classification Report:\n", classification_report(LABELS, PRED_LABELS))
print("Confusion Matrix:\n", confusion_matrix(LABELS, PRED_LABELS))