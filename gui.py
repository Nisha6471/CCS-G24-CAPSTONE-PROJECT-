import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import customtkinter as ctk
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import mysql.connector
from mysql.connector import Error
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64


# Connect to MySQL Database
def connect_db():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',  # replace with your MySQL username
            password='2003',  # replace with your MySQL password
            database='image_data_db'
        )
        if conn.is_connected():
            return conn
    except Error as e:
        print(f"Error: {e}")
        return None


# Generate a key using PBKDF2HMAC (this can be securely stored in your application settings or a file)
def generate_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(password.encode())


# Save the key securely (in this case, store it as base64 encoded in the database for later retrieval)
def save_key_to_db(salt: bytes, key: bytes):
    conn = connect_db()
    if conn is not None:
        cursor = conn.cursor()
        try:
            # Storing the salt and base64 encoded key
            key_encoded = base64.b64encode(key).decode('utf-8')
            query = "INSERT INTO encryption_keys (salt, key) VALUES (%s, %s)"
            cursor.execute(query, (base64.b64encode(salt).decode('utf-8'), key_encoded))
            conn.commit()
        except Error as e:
            print(f"Error saving key to DB: {e}")
        finally:
            cursor.close()
            conn.close()


# Retrieve key from DB
def retrieve_key_from_db():
    conn = connect_db()
    if conn is not None:
        cursor = conn.cursor()
        try:
            query = "SELECT salt, key FROM encryption_keys ORDER BY id DESC LIMIT 1"
            cursor.execute(query)
            result = cursor.fetchone()
            if result:
                salt = base64.b64decode(result[0])
                key = base64.b64decode(result[1])
                return salt, key
        except Error as e:
            print(f"Error retrieving key from DB: {e}")
        finally:
            cursor.close()
            conn.close()
    return None, None


# Insert image data, IV, and timestamp into MySQL
def insert_image_data(image_data, iv, salt):
    conn = connect_db()
    if conn is not None:
        cursor = conn.cursor()
        try:
            query = "INSERT INTO images (image_data, iv, salt) VALUES (%s, %s, %s)"
            cursor.execute(query, (image_data, iv, salt))
            conn.commit()
        except Error as e:
            print(f"Error inserting data: {e}")
        finally:
            cursor.close()
            conn.close()

# Function to convert IV to a human-readable format (base64 encoding)
def iv_to_base64(iv):
    return base64.b64encode(iv).decode('utf-8')

# Updated capture methods to show IV pop-up notification
def capture_face(self):
    self.face_image = self.capture_image()
    if self.face_image is not None:
        self.update_preview(self.preview_canvas1, self.face_image)
        self.capture_button1.configure(state="disabled")
        self.capture_button2.configure(state="normal")

        # Encrypt the captured face image and get the IV
        encrypted_data, iv, salt = encrypt_image(self.face_image, password="secure_password")

        # Show IV in a pop-up message box
        iv_base64 = iv_to_base64(iv)
        messagebox.showinfo("Capture Successful", f"Face Image Captured!\nIV (Base64): {iv_base64}")

        # Optionally, insert image and IV data into DB
        insert_image_data(encrypted_data, iv, salt)

def capture_iris(self):
    self.iris_image = self.capture_image()
    if self.iris_image is not None:
        self.update_preview(self.preview_canvas2, self.iris_image)
        self.capture_button2.configure(state="disabled")
        self.process_button.configure(state="normal")

        # Encrypt the captured iris image and get the IV
        encrypted_data, iv, salt = encrypt_image(self.iris_image, password="secure_password")

        # Show IV in a pop-up message box
        iv_base64 = iv_to_base64(iv)
        messagebox.showinfo("Capture Successful", f"Iris Image Captured!\nIV (Base64): {iv_base64}")

        # Optionally, insert image and IV data into DB
        insert_image_data(encrypted_data, iv, salt)

class FullScreenApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Feature Map Visualization")
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}")  # Fullscreen window
        self.state('zoomed')  # Maximize window

        # Initialize MobileNetV2 for visualization
        self.feature_visualizer, self.layer_names = self.initialize_feature_visualizer()

        # Variables to store images
        self.face_image = None
        self.iris_image = None

        # Layout setup
        self.setup_ui()

    def initialize_feature_visualizer(self):
        base_model = MobileNetV2(weights="imagenet", include_top=False)
        layer_names = ["Conv1", "block_1_expand", "block_3_expand", "block_6_expand"]
        intermediate_model = Model(
            inputs=base_model.input,
            outputs=[base_model.get_layer(name).output for name in layer_names]
        )
        return intermediate_model, layer_names

    def setup_ui(self):
        # Main Scrollable Container
        self.scroll_canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbars
        self.vertical_scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        self.vertical_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.horizontal_scrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.scroll_canvas.xview)
        self.horizontal_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.scroll_canvas.configure(yscrollcommand=self.vertical_scrollbar.set,
                                     xscrollcommand=self.horizontal_scrollbar.set)

        # Bind scrolling
        self.scroll_canvas.bind('<Configure>', lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))

        # Frame inside the Canvas
        self.scroll_frame = tk.Frame(self.scroll_canvas, bg="white")
        self.scroll_canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")

        # Left and Right Frames inside Scrollable Area
        self.left_frame = ctk.CTkFrame(self.scroll_frame, width=600, corner_radius=10)
        self.left_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nw")

        self.right_frame = ctk.CTkFrame(self.scroll_frame, width=1000, corner_radius=10)
        self.right_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")  # Ensure it expands fully


        # Camera and Previews in Left Frame
        self.setup_left_frame()

        # Output Section in Right Frame
        self.setup_right_frame()

    def setup_left_frame(self):
        self.preview_label1 = ctk.CTkLabel(self.left_frame, text="Face Preview", font=("Arial", 16))
        self.preview_label1.pack(pady=5)

        self.preview_canvas1 = tk.Canvas(self.left_frame, width=300, height=300, bg="gray")
        self.preview_canvas1.pack(pady=5)

        self.preview_label2 = ctk.CTkLabel(self.left_frame, text="Iris Preview", font=("Arial", 16))
        self.preview_label2.pack(pady=5)

        self.preview_canvas2 = tk.Canvas(self.left_frame, width=300, height=300, bg="gray")
        self.preview_canvas2.pack(pady=5)

        self.capture_button1 = ctk.CTkButton(self.left_frame, text="Capture Face", command=self.capture_face)
        self.capture_button1.pack(pady=5)

        self.capture_button2 = ctk.CTkButton(self.left_frame, text="Capture Iris", command=self.capture_iris, state="disabled")
        self.capture_button2.pack(pady=5)

        self.process_button = ctk.CTkButton(self.left_frame, text="Process & Visualize", command=self.process_and_visualize, state="disabled")
        self.process_button.pack(pady=5)

    def setup_right_frame(self):
        # Label for feature maps and results
        self.result_label = ctk.CTkLabel(self.right_frame, text="Feature Maps and Results", font=("Arial", 16))
        self.result_label.pack(pady=10, fill=tk.X)

        # Create a frame to hold the canvas and scrollbar together
        self.scroll_frame = tk.Frame(self.right_frame, width=1400)  # Fixed width for the scroll_frame
        self.scroll_frame.pack(fill=tk.BOTH, expand=True)

        # Create a canvas to make the result_container scrollable
        self.canvas = ctk.CTkCanvas(self.scroll_frame, bg="white", width=1400)  # Fixed width for the canvas
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a vertical scrollbar to the canvas
        self.scrollbar = ctk.CTkScrollbar(self.scroll_frame, orientation="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill="y")

        # Configure the canvas to use the scrollbar
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Create the result_container frame inside the canvas
        self.result_container = tk.Frame(self.canvas, bg="white")
        self.canvas.create_window((0, 0), window=self.result_container, anchor="nw")

        # Update the scrollregion when result_container changes size
        self.result_container.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))


    def capture_image(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Unable to access camera.")
            return None
        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image.")
            return None
        return frame

    def capture_face(self):
        self.face_image = self.capture_image()
        if self.face_image is not None:
            self.update_preview(self.preview_canvas1, self.face_image)
            self.capture_button1.configure(state="disabled")
            self.capture_button2.configure(state="normal")

    def capture_iris(self):
        self.iris_image = self.capture_image()
        if self.iris_image is not None:
            self.update_preview(self.preview_canvas2, self.iris_image)
            self.capture_button2.configure(state="disabled")
            self.process_button.configure(state="normal")

    def update_preview(self, canvas, image):
        image_resized = cv2.resize(image, (300, 300))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        tk_image = ImageTk.PhotoImage(pil_image)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)
        canvas.image = tk_image

    def preprocess_image(self, image):
        image_resized = cv2.resize(image, (224, 224))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_preprocessed = preprocess_input(image_rgb.astype(np.float32))
        return np.expand_dims(image_preprocessed, axis=0)

    def process_and_visualize(self):
        face_preprocessed = self.preprocess_image(self.face_image)
        iris_preprocessed = self.preprocess_image(self.iris_image)

        face_features = self.feature_visualizer.predict(face_preprocessed)
        iris_features = self.feature_visualizer.predict(iris_preprocessed)

        self.visualize_features(face_features, "Face")
        self.visualize_features(iris_features, "Iris")

    def visualize_features(self, feature_maps, image_title):
        for layer_name, layer_output in zip(self.layer_names, feature_maps):
            num_filters = layer_output.shape[-1]
            cols = 8
            rows = min((num_filters + cols - 1) // cols, 4)
            fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
            for i in range(rows * cols):
                if i < num_filters:
                    ax = axes[i // cols, i % cols]
                    ax.imshow(layer_output[0, :, :, i], cmap='viridis')
                    ax.axis('off')
                else:
                    axes[i // cols, i % cols].axis('off')
            plt.suptitle(f"{image_title} - Feature Maps ({layer_name})", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            canvas = FigureCanvasTkAgg(fig, master=self.result_container)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, pady=5)

            self.result_container.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))



if __name__ == "__main__":
    app = FullScreenApp()
    app.mainloop()
