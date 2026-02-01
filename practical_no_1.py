## **Practical 1**
"""Aim:
Write program to demonstrate the following aspects of signal processing on suitable data.
1.	Upsampling and downsampling on Image/speech signal
2.	Fast Fourier Transform to compute DFT

**1) Upsampling and downsampling on Image/speech signal.**
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2

"""*    Function to upsample an image

"""

def upsample_image(image, factor):
    return ndimage.zoom(image, factor)

"""* Function to downsample an image"""

def downsample_image(image, factor):
    return image[::factor, ::factor]

img_path = "image.png"
original_imge = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
upsampled_imge = upsample_image(original_imge, 2)

downsampled_imge = downsample_image(upsampled_imge, 2)
plt.figure(figsize=(20, 10))
plt.subplot(1, 3, 1)
plt.imshow(original_imge, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(upsampled_imge, cmap='gray')
plt.title('Upsampled Image')

plt.subplot(1, 3, 3)
plt.imshow(downsampled_imge, cmap='gray')
plt.title('Downsampled Image')

plt.tight_layout()
plt.show()

"""**2. Fast Fourier Transform to compute DFT.**"""

image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

"""* Compute DFT"""

dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

"""* Compute magnitude spectrum"""

magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum (DFT)')
plt.axis('off')

plt.tight_layout()
plt.show()