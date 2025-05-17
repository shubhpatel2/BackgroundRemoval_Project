import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from utils1.mask_utils import apply_mask_transparent

# --- Configuration ---
INPUT_FOLDER = 'Data/Test_image'
OUTPUT_FOLDER = 'output/'
MODEL_PATH = 'pretrained_unet_best_model.keras' 

# Create output directory if not exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- Load the full model directly ---
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# --- Loop through all images ---
for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(INPUT_FOLDER, filename)
        original = Image.open(image_path).convert('RGB')
        original_size = original.size

        # Preprocess
        resized_image = original.resize((256, 256))
        input_array = np.array(resized_image) / 255.0
        input_tensor = np.expand_dims(input_array, axis=0)

        # Predict
        predicted_mask = model.predict(input_tensor, verbose=0)[0]

        # Optional visualization
        plt.imshow(predicted_mask.squeeze(), cmap='gray')
        plt.title(f"Predicted Mask - {filename}")
        plt.axis('off')
        plt.show()

        # Binarize and resize to original
        binary_mask = (predicted_mask > 0.5).astype(np.uint8).squeeze()
        resized_mask = cv2.resize(binary_mask, original_size, interpolation=cv2.INTER_NEAREST)

        # Apply transparent mask and save
        rgba_image = apply_mask_transparent(np.array(original), resized_mask)
        output_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}_transparent.png")
        Image.fromarray(rgba_image).save(output_path)

        print(f"✅ Saved: {output_path}")
