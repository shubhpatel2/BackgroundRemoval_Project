import numpy as np
import cv2
from PIL import Image

def apply_mask_transparent(image, mask):
   
    if mask.ndim == 3:
        mask = mask.squeeze()
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D after squeezing.")

    # Ensure mask is 0 or 255 for alpha channel
    alpha = (mask * 255).astype(np.uint8)

    # Ensure image is uint8
    image = image.astype(np.uint8)
    
    # Convert to RGBA and assign alpha channel
    rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = alpha
    return rgba
