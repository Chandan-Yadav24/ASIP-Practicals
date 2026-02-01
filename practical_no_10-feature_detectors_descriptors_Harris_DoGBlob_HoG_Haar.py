"""# **Practical No 10**

Write the program to extract image features by implementing methods like corner and
blob detectors, HoG and Haar features
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

# Corner Detection
corners = cv2.goodFeaturesToTrack(original_imge, 100, 0.01, 10)
corners = corners.reshape(-1, 2)
corner_image = original_imge.copy()
for corner in corners:
  x, y = corner
  cv2.circle(corner_image, (int(x), int(y)), 5, (0, 255, 0),-1)

# Blob Detection
detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(original_imge)
blob_image = cv2.drawKeypoints(original_imge, keypoints, None, (0, 0, 255),
cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Haar Features (Face Detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(original_imge, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
face_image = original_imge.copy()
for (x, y, w, h) in faces:
  cv2.rectangle(face_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display results using matplotlib
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 4, 1)
plt.imshow(original_imge, cmap='gray') # Display as grayscale
plt.title('Original Image')
plt.axis('off')

plt.figure(figsize=(12, 6))
plt.subplot(1, 4, 2)
plt.imshow(cv2.cvtColor(corner_image, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for colored circle
plt.title('Corner Detection')
plt.axis('off')
# Blob Detection Image
plt.subplot(1, 4, 3)
plt.imshow(cv2.cvtColor(blob_image, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for colored keypoints
plt.title('Blob Detection')
plt.axis('off')
# Face Detection Image
plt.subplot(1, 4, 4)
plt.imshow(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)) # Convert BGR to RGB for colored rectangles
plt.title('Face Detection')
plt.axis('off')
plt.tight_layout()
plt.show()