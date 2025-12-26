# Deep Learning Background Removal Tool

A deep learning-based background removal tool using U-Net for pixel-wise semantic segmentation. This project involved implementing the architecture in PyTorch on a custom dataset of 5,000+ images, predicting object masks to output images with transparent backgrounds.

Model: U-Net (Convolutional Neural Network) for semantic segmentation.

Objective: Automatically remove the background from an image and make it transparent.

Key Performance:

Achieved a mean Intersection over Union (IoU) score of 0.89 on validation data.

Achieved a pixel-wise accuracy of 96%.

The trained model is packaged as a reusable Python module for easy integration.

Input: RGB images (JPEG/PNG format).

Output: RGBA images with transparent background (PNG format).

Dataset: Trained on the DUTS-TE dataset (5,000+ images) and corresponding binary masks.

Download: DUTS Saliency Detection Dataset

Preprocessing: Applied extensive augmentation (flips, rotations, color jitter) to enhance robustness against diverse backgrounds.

Frameworks: PyTorch, OpenCV, NumPy, PIL.

Training: Model trained using binary cross-entropy (BCE) loss.

Prediction: Single image or batch input supported; output saved with transparency
