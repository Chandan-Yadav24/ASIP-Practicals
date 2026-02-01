"""# **Practical No 8**

Write a program to Apply edge detection techniques such as Sobel and Canny to extract
meaningful information from the given image samples
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2

image_path = 'image.png'
original_imge = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Sobel edge detection
sobel_x = cv2.Sobel(original_imge, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(original_imge, cv2.CV_64F, 0, 1, ksize=3)

sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

sobel_direction = np.arctan2(sobel_y,sobel_x)
# Apply Canny edge detection
canny_edges = cv2.Canny(original_imge, 50, 150)

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.imshow(original_imge, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 3, 2)
plt.imshow(np.abs(sobel_x), cmap='gray')
plt.title('Sobel X')

plt.subplot(2, 3, 3)
plt.imshow(np.abs(sobel_y), cmap='gray')
plt.title('Sobel Y')

plt.tight_layout()
plt.show()
