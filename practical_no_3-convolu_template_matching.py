"""# **Practical No 3**

Write program to demonstrate the following aspects of signal on sound/image data
1. Convolution operation
2. Template Matching
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

"""Function to perform convolution operation"""

def convolution(image, kernel):
    return cv2.filter2D(image, -1, kernel)

"""Function to perform template matching"""

def template_matching(image, template):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    return result

"""Load the image and template"""

image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

"""Define a simple edge detection kernel for convolution"""

edge_detection_kernel = np.array([[-1, -1, -1],
[-1, 8, -1],
[-1, -1, -1]], dtype=np.float32)

# Convolution operation with the edge detection kernel
convolved_image = convolution(image, edge_detection_kernel)

# Template Matching
matched_result = template_matching(image, template)

# Display the original image, convolved image, and template matching result
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(132)
plt.imshow(convolved_image, cmap='gray')
plt.title('Convolved Image (Edge Detection)')

plt.subplot(133)
plt.imshow(matched_result, cmap='jet') # Use 'jet' colormap for better visibility
plt.title('Template Matching Result')

plt.tight_layout()
plt.show()
