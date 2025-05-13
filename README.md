# CNN_Model_BackgroundRemoval
A deep learning-based background removal tool using U-Net for image segmentation. Trained on paired images and masks, it predicts object masks and outputs images with transparent backgrounds

Model: U-Net (Convolutional Neural Network) for semantic segmentation

Objective: Automatically remove the background from an image and make it transparent

Input: RGB images (JPEG/PNG format)

Output: RGBA images with transparent background (PNG format)

Dataset: DUTS-TE images and corresponding binary masks.U can download Training datasets from https://www.kaggle.com/datasets/balraj98/duts-saliency-detection-dataset?select=DUTS-TE

Frameworks: TensorFlow, Keras, OpenCV, NumPy, PIL

Training: Model trained using binary cross-entropy with accuracy and IoU metrics

Prediction: Single image or batch input supported; output saved with transparency
