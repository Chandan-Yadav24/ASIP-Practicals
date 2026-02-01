"""# **PRACTICAL NO. 6**

The aim of this program is to demonstrate two common techniques used for noise reduction in
images: linear smoothing and nonlinear smoothing.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

image_path = 'image.png'
original_imge = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded successfully
if original_imge is None:
    print("Error: Image not found or could not be loaded.")
    exit()

# Add Gaussian noise to the image
noise = np.random.normal(0, 25, original_imge.shape).astype('uint8')
noisy_image = cv2.add(original_imge, noise)

# Function to apply linear (Gaussian blur) smoothing

def linear_smoothing(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# Function to apply nonlinear (median filter) smoothing
def nonlinear_smoothing(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

# Apply linear smoothing (Gaussian blur)
linear_smoothed_image = linear_smoothing(noisy_image, kernel_size=5)

# Apply nonlinear smoothing (median filter)
nonlinear_smoothed_image = nonlinear_smoothing(noisy_image, kernel_size=5)

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.imshow(original_imge, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')

plt.tight_layout()
plt.show()

# Display the results
plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.imshow(linear_smoothed_image, cmap='gray')
plt.title('Linear Smoothing (Gaussian Blur)')
plt.subplot(2, 3, 2)
plt.imshow(nonlinear_smoothed_image, cmap='gray')
plt.title('Nonlinear Smoothing (Median Filter)')
plt.tight_layout()
plt.show()
