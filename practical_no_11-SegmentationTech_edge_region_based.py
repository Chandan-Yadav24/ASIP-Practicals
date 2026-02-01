"""# **Practical No 11**

The aim of this program is to apply segmentation techniques to detect lines, circles, and other shapes/objects in an image using both edge-based and region-based segmentation methods.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
import os

image_path = 'image.png'
original_imge = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def display_images(images, titles):
    plt.figure(figsize=(12, 6))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

def edge_based_segmentation(image):
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
    lines_image = np.copy(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_image, (x1, y1), (x2, y2), 255, 2)
    return [edges, lines_image], ['Canny Edges', 'Detected Lines']

def region_based_segmentation(image):
    # Thresholding to create a binary image
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # Find contours of objects
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours on a black image
    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours,-1, 255, 2)
    return [binary_image, contour_image], ['Binary Image', 'Detected Contours']

edge_segmentation_result, edge_segmentation_titles = edge_based_segmentation(original_imge)
# Apply region-based segmentation
region_segmentation_result, region_segmentation_titles = region_based_segmentation(original_imge)
# Display the results
display_images(edge_segmentation_result, edge_segmentation_titles)

display_images(region_segmentation_result, region_segmentation_titles)