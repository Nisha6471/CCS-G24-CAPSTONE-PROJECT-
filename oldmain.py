import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from hashlib import sha256
import base64
from sklearn.decomposition import PCA

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

# AES decryption
def aes_decrypt(iv, ciphertext, key):
    iv = base64.b64decode(iv)  # Decode IV from base64
    ciphertext = base64.b64decode(ciphertext)  # Decode ciphertext from base64
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_data = unpad(cipher.decrypt(ciphertext), AES.block_size)  # Remove padding
    return decrypted_data.decode('utf-8')

# Capture image from the camera
def capture_image(window_name="Capture Image"):
    cap = cv2.VideoCapture(0)
    print(f"Press 'c' to capture the image for {window_name}.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            cap.release()
            cv2.destroyAllWindows()
            return frame
        elif key == ord('q'):
            print("Camera feed closed.")
            cap.release()
            cv2.destroyAllWindows()
            return None

# Preprocess image for MobileNetV2
def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_input(image.astype(np.float32))  # Normalize for MobileNetV2
    return np.expand_dims(image, axis=0)

# Visualize feature maps
def visualize_features(intermediate_outputs, layer_names, image_title):
    for layer_name, feature_maps in zip(layer_names, intermediate_outputs):
        print(f"Visualizing features from layer: {layer_name} for {image_title}")
        num_filters = feature_maps.shape[-1]
        cols = 8
        rows = min((num_filters + cols - 1) // cols, 4)  # Max 4 rows
        fig, axes = plt.subplots(rows, cols, figsize=(12, 6))
        for i in range(rows * cols):
            if i < num_filters:
                ax = axes[i // cols, i % cols]
                ax.imshow(feature_maps[0, :, :, i], cmap="viridis")
                ax.axis("off")
            else:
                axes[i // cols, i % cols].axis("off")
        plt.suptitle(f"Features from {layer_name} ({image_title})")
        plt.show()

# Generate biometric key from features
def generate_biometric_key(features):
    features_str = ','.join(map(str, features.flatten()))
    hash_key = sha256(features_str.encode()).digest()
    return hash_key[:16]  # Return a 16-byte key suitable for AES

# AES encryption
def aes_encrypt(data, key):
    cipher = AES.new(key, AES.MODE_CBC)  # CBC mode requires initialization vector (IV)
    ct_bytes = cipher.encrypt(pad(data.encode(), AES.block_size))
    iv = base64.b64encode(cipher.iv).decode('utf-8')  # Encode IV to base64
    ct = base64.b64encode(ct_bytes).decode('utf-8')  # Encode ciphertext to base64
    return iv, ct

# Visualize feature data if it's not image data
def visualize_feature_data(decrypted_data):
    # Convert decrypted data to a numpy array of floats
    feature_data = np.array([float(i) for i in decrypted_data.split(',')])
    
    # Optionally reduce dimensions using PCA (or any dimensionality reduction technique)
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(feature_data.reshape(-1, 1))  # Reshape for PCA

    # Plot the reduced data
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
    plt.title("Feature Data Visualization")
    plt.show()

def save_decrypted_image(decrypted_data, image_shape):
    try:
        # Convert the decrypted data back to a numpy array
        decrypted_array = np.frombuffer(decrypted_data.encode(), dtype=np.uint8)
        
        # Check if the decrypted data size matches the expected image size or feature size
        expected_size = np.prod(image_shape)
        if decrypted_array.size != expected_size:
            # Handle feature data, visualize it
            print(f"Decrypted data size {decrypted_array.size} does not match expected image size {expected_size}.")
            visualize_feature_data(decrypted_data)  # Visualize features
            return
        else:
            # If the size matches, proceed to reshape it as an image
            image = decrypted_array.reshape(image_shape)
            # Save the image
            image_filename = "decrypted_image.png"
            cv2.imwrite(image_filename, image)
            print(f"Image saved as {image_filename}")
        
    except Exception as e:
        print(f"Error saving decrypted image: {e}")

# Main workflow
def main():
    print("Initializing feature extractor...")
    feature_visualizer, layer_names = initialize_feature_visualizer()

    # Capture face image
    print("Step 1: Capture Face Image")
    face_image = capture_image("Face Capture")
    if face_image is None:
        print("Face capture failed. Exiting...")
        return

    # Capture iris image
    print("Step 2: Capture Iris Image")
    iris_image = capture_image("Iris Capture")
    if iris_image is None:
        print("Iris capture failed. Exiting...")
        return

    # Preprocess the images
    print("Preprocessing the captured images...")
    face_preprocessed = preprocess_image(face_image)
    iris_preprocessed = preprocess_image(iris_image)

    # Get intermediate outputs for the face
    print("Processing the face image through MobileNetV2...")
    face_intermediate_outputs = feature_visualizer.predict(face_preprocessed)

    # Visualize features for the face image
    print("Visualizing features for the face image...")
    visualize_features(face_intermediate_outputs, layer_names, "Face Image")

    # Get intermediate outputs for the iris
    print("Processing the iris image through MobileNetV2...")
    iris_intermediate_outputs = feature_visualizer.predict(iris_preprocessed)

    # Visualize features for the iris image
    print("Visualizing features for the iris image...")
    visualize_features(iris_intermediate_outputs, layer_names, "Iris Image")

    # Combine features from both images and generate biometric key
    print("Generating biometric key...")
    combined_features = np.concatenate((face_intermediate_outputs[0], iris_intermediate_outputs[0]), axis=None)
    biometric_key = generate_biometric_key(combined_features)
    print("Biometric Key Generated (Hex):", biometric_key.hex())

    # Encrypt combined features using AES
    print("Encrypting the features...")
    combined_features_str = ','.join(map(str, combined_features))
    iv, encrypted_data = aes_encrypt(combined_features_str, biometric_key)

    # Save the encrypted data to a file
    with open("encrypted_data.txt", "w") as enc_file:
        enc_file.write(encrypted_data)

    # Display the IV
    print(f"IV (Base64 encoded): {iv}")

    # Ask user if they want to decrypt the data
    decrypt_choice = input("Do you want to decrypt the data? (yes/no): ")
    if decrypt_choice.lower() == 'yes':
        # Prompt user to input the IV and ciphertext for decryption
        iv_input = input("Enter the IV: ")
        ciphertext_input = open("encrypted_data.txt", "r").read()

        # Decrypt the data
        decrypted_data = aes_decrypt(iv_input, ciphertext_input, biometric_key)
        print("Decryption complete. Decrypted data saved to 'decrypted_data.txt'.")

        # Save the decrypted data to a file
        with open("decrypted_data.txt", "w") as dec_file:
            dec_file.write(decrypted_data)

        # Define the expected image shape
        image_shape = (224, 224, 3)  # Adjust to match the size of your processed images
        save_decrypted_image(decrypted_data, image_shape)

if __name__ == "__main__":
    main()