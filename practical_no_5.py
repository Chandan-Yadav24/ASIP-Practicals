
"""# **PRACTICAL NO. 5**

Aim: To apply gradient and Laplacian for image enhancement
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Load an image
image_path = "image.png"
original_imge = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Gradient (Sobel) operation
gradient_x = cv2.Sobel(original_imge, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(original_imge, cv2.CV_64F, 0, 1, ksize=3)

# Combine the gradients to get the magnitude and direction
gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
gradient_direction = np.arctan2(gradient_y,gradient_x)

# Laplacian
laplacian_image = cv2.Laplacian(original_imge, cv2.CV_64F)

# Display the results
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(original_imge, cmap='gray')
plt.title('Original Image')
plt.subplot(2, 3, 2)
plt.imshow(gradient_x, cmap='gray')
plt.title('Gradient X')
plt.subplot(2, 3, 3)
plt.imshow(gradient_y, cmap='gray')
plt.title('Gradient Y')
plt.subplot(2, 3, 4)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Gradient Magnitude')
plt.subplot(2, 3, 5)
plt.imshow(gradient_direction, cmap='gray')
plt.title('Laplacian direction')
plt.subplot(2, 3, 6)
plt.imshow(laplacian_image, cmap='gray')
plt.title('Laplacian')
plt.tight_layout()
plt.show()
