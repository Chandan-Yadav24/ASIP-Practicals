"""# **Practical No 9**

Write the program to implement various morphological image processing techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
import os

image_path = 'image.png'
original_imge = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

image_path = 'image.png'
if not os.path.exists(image_path):
  raise FileNotFoundError(f"File not found: {image_path}")

original_imge = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if original_imge is None:
  raise ValueError("Image failed to load")

def display_images(images, titles):
  plt.figure(figsize=(12, 8))
  for i in range(len(images)):
    plt.subplot(1, len(images), i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
  plt.show()

def apply_morphological_operations(image):
  kernel = np.ones((5, 5), np.uint8)
  erosion = cv2.erode(image, kernel, iterations=1)
  dilation = cv2.dilate(image, kernel, iterations=1)
  opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
  closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
  return [image, erosion, dilation, opening, closing]

morph_images = apply_morphological_operations(original_imge)
titles = ["Original", "Erosion", "Dilation", "Opening", "Closing"]
display_images(morph_images[:3], titles[:3])

display_images(morph_images[3:], titles[3:])