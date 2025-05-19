from flask import Flask, request, jsonify
import numpy as np
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, InceptionV3
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# ✅ Load the trained SVM model
svm_model = joblib.load("P:/OneDrive/Desktop/SignSignify/App2/signsignify/asset/python/ISL_SVM_Model.pkl")

# ✅ Load the scalers for feature normalization
scaler_resnet = joblib.load("P:/OneDrive/Desktop/SignSignify/App2/signsignify/asset/python/scaler_resnet.pkl")
scaler_inception = joblib.load("P:/OneDrive/Desktop/SignSignify/App2/signsignify/asset/python/scaler_inception.pkl")

# ✅ Load feature extractors (ResNet50 & InceptionV3)
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
inception_model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")

# ✅ ISL Label Mapping
label_map = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I",
    19: "J", 20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R",
    28: "S", 29: "T", 30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z"
}

# ✅ Preprocessing function (Resizes images for models)
def preprocess_image(image):
    img_resnet = cv2.resize(image, (224, 224))  # ResNet50 input size
    img_inception = cv2.resize(image, (299, 299))  # InceptionV3 input size

    img_resnet = img_to_array(img_resnet) / 255.0
    img_inception = img_to_array(img_inception) / 255.0

    img_resnet = np.expand_dims(img_resnet, axis=0)
    img_inception = np.expand_dims(img_inception, axis=0)

    return img_resnet, img_inception

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 🛑 Check if image is in request
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        # ✅ Preprocess Image
        img_resnet, img_inception = preprocess_image(image)

        # ✅ Extract Features using CNNs
        features_resnet = resnet_model.predict(img_resnet)
        features_inception = inception_model.predict(img_inception)

        # ✅ Normalize Features
        features_resnet = scaler_resnet.transform(features_resnet)
        features_inception = scaler_inception.transform(features_inception)

        # ✅ Combine Features
        features_combined = np.concatenate((features_resnet, features_inception), axis=1)

        # ✅ Predict Sign using SVM
        pred_label = svm_model.predict(features_combined)[0]
        predicted_sign = label_map.get(pred_label, "Unknown")

        return jsonify({"prediction": predicted_sign})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Run Flask API
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)