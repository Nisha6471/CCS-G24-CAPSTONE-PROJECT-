from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from hashlib import sha256
import base64
from sklearn.decomposition import PCA
from PIL import Image
import io
import matplotlib.pyplot as plt

app = Flask(__name__)

# Initialize MobileNetV2 with intermediate layers
def initialize_feature_visualizer():
    base_model = MobileNetV2(weights="imagenet", include_top=False)
    layer_names = [
        "Conv1",               # Initial convolutional layer (edges and textures)
        "block_1_expand",      # Early features (basic patterns)
        "block_3_expand",      # Mid-level features (shapes)
        "block_6_expand",      # Deeper features (complex patterns)
    ]
    intermediate_model = Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(name).output for name in layer_names]
    )
    return intermediate_model, layer_names

# AES encryption
def aes_encrypt(data, key):
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(data.encode(), AES.block_size))
    iv = base64.b64encode(cipher.iv).decode('utf-8')
    ct = base64.b64encode(ct_bytes).decode('utf-8')
    return iv, ct

# Generate a biometric key from features
def generate_biometric_key(features):
    features_str = ','.join(map(str, features.flatten()))
    hash_key = sha256(features_str.encode()).digest()
    return hash_key[:16]  # 16-byte AES key

# Convert feature maps to images
def feature_map_to_image(feature_map):
    feature_map = np.squeeze(feature_map)
    feature_map = feature_map[0] if feature_map.ndim == 3 else feature_map
    feature_map -= np.min(feature_map)
    feature_map /= np.max(feature_map)
    colored_feature_map = plt.cm.viridis(feature_map)[:, :, :3] * 255
    image = Image.fromarray(colored_feature_map.astype(np.uint8))
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

# Upload and process images
@app.route('/upload', methods=['POST'])
def upload_images():
    image1 = request.files['image1']
    image2 = request.files['image2']
    img1 = cv2.imdecode(np.frombuffer(image1.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(image2.read(), np.uint8), cv2.IMREAD_COLOR)

    feature_visualizer, layer_names = initialize_feature_visualizer()
    preprocess = lambda img: preprocess_input(cv2.resize(img, (224, 224)).astype(np.float32))
    img1_preprocessed, img2_preprocessed = map(preprocess, [img1, img2])
    
    img1_features = feature_visualizer.predict(np.expand_dims(img1_preprocessed, axis=0))
    img2_features = feature_visualizer.predict(np.expand_dims(img2_preprocessed, axis=0))

    feature_map_images = {
        f"img1_{layer}": feature_map_to_image(img1_features[idx][0])
        for idx, layer in enumerate(layer_names)
    }
    feature_map_images.update({
        f"img2_{layer}": feature_map_to_image(img2_features[idx][0])
        for idx, layer in enumerate(layer_names)
    })

    combined_features = np.concatenate((img1_features[0], img2_features[0]), axis=None)
    biometric_key = generate_biometric_key(combined_features)
    combined_features_str = ','.join(map(str, combined_features))
    iv, encrypted_data = aes_encrypt(combined_features_str, biometric_key)

    return jsonify({
        "encrypted_data": encrypted_data,
        "iv": iv,
        "feature_maps": feature_map_images
    })

# Main route
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
