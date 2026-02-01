
"""# **Practical No 7**

Aim: The aim of this program is to demonstrate various image enhancement techniques using image
derivatives, including smoothing, sharpening, and unsharp masking, to generate images suitable for
specific application requirements.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# load image in grayscale
image_path = 'image.png'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# -------- FILTER FUNCTIONS --------
def apply_smoothing(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_sharpening(image):
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])
    return cv2.filter2D(image, -1, kernel)

def apply_unsharp_masking(image, sigma=4.0, strength=5.0):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened


# -------- APPLY FILTERS --------
smoothed_image = apply_smoothing(original_image, kernel_size=17)
sharpened_image = apply_sharpening(original_image)
unsharp_masked_image = apply_unsharp_masking(original_image)


# -------- DISPLAY OUTPUT --------
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(smoothed_image, cmap='gray')
plt.title('Smoothing')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(sharpened_image, cmap='gray')
plt.title('Sharpening')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(unsharp_masked_image, cmap='gray')
plt.title('Unsharp Masking ')
plt.axis('off')

plt.tight_layout()
plt.show()