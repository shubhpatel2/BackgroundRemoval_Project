import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class BackgroundRemovalDataset:
    def __init__(self, image_dir='Data/DUTS-TE-Image', mask_dir='Data/DUTS-TE-Masks', image_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.images = []
        self.masks = []

    def load_data(self):
        image_files = sorted(os.listdir(self.image_dir))
        mask_files = sorted(os.listdir(self.mask_dir))

        for img_file, mask_file in zip(image_files, mask_files):
            img_path = os.path.join(self.image_dir, img_file)
            mask_path = os.path.join(self.mask_dir, mask_file)

            # Load and resize image
            img = load_img(img_path, target_size=self.image_size)
            img = img_to_array(img) / 255.0  # Normalize to [0, 1]

            # Load and resize mask
            mask = load_img(mask_path, target_size=self.image_size, color_mode='grayscale')
            mask = img_to_array(mask) / 255.0
            mask = (mask > 0.5).astype(np.float32)  # Binarize

            self.images.append(img)
            self.masks.append(mask)

        return np.array(self.images), np.array(self.masks)
        
        print("✅ datasets.py loaded!")
    

    def get_splits(self, test_size=0.2, random_state=42):
        X, y = self.load_data()
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
