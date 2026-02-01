"""# **Practical No 4**

Aim: Write program to implement point/pixel intensity transformations such as:
1. Log and Power-law transformations
2. Contrast adjustments
3. Histogram equalization
4. Thresholding, and halftoning operations
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

"""**1. Log and Power-law (Gamma) transformations**"""

def Log_tranform(img_input,C ):
    Log_image = C * np.log((1 + img_input))
    return Log_image

def power_law_T(img_input,gamma):
    power_law_img = 1 * (img_input**gamma)
    return power_law_img

"""**2. Contrast adjustment (contrast stretching)**"""

def contrast_st(img_input):
    min_val = np.min(img_input)
    max_val = np.max(img_input)
    for i in range(img_input.shape[0]):
        for pix in range(img_input.shape[1]):
            img_input[i][pix] = ((img_input[i][pix] - min_val) / (max_val - min_val)) * 255

    stretched_image = img_input
    return stretched_image

"""**3. Histogram computation and Histogram equalization**"""

def calculate_histrogramgram(image):
    histrogramgram = np.zeros(256, dtype=int)
    for row in range(image.shape[0]):
        for pixel_value in range(image.shape[1]):
            histrogramgram[ image[row][pixel_value] ] += 1
    return histrogramgram

def histrogramgram_equalization(image):
    # Calculate the histrogramgram
    row, col = image.shape
    histrogramgram = calculate_histrogramgram(image)
    # Calculate the cumulative distribution function (CDF)
    pdf = histrogramgram / (row * col)
    cdf = np.cumsum(pdf)

    # Normalize CDF to 0-255
    h_eq = np.round(cdf * 255).astype(np.uint8)
    return h_eq

"""**4.Thresholding and Halftoning operations**"""

def thresolding(image,t : int):
    for i in range(image.shape[0]):
        for pix in range(image.shape[1]):
            if image[i][pix] <= t:
                image[i][pix] = 0
            else:
                image[i][pix] = 255
    return image

def halftone(image):
    # Define a dithering matrix
    dither_matrix = np.array([[0, 8, 2, 10],
                              [12, 4, 14, 6],
                              [3, 11, 1, 9],
                              [15, 7, 13, 5]])
    # Load the image in grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Initialize halftoned image
    halftoned_image = np.zeros_like(image_gray)
    # Apply halftoning
    for i in range(image_gray.shape[0]):
        for j in range(image_gray.shape[1]):
            if image_gray[i, j] > dither_matrix[i % 4, j % 4]:
                halftoned_image[i, j] = 255
    return halftoned_image

"""**Display images**"""

def display_histograms(hist_org, hist_eq):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.bar(range(256), hist_org)
    plt.title('Histogram of Original Image')
    plt.subplot(1, 2, 2)
    plt.bar(range(256), hist_eq)
    plt.title('Histogram of Histogram Equalized Image')
    plt.show()

def display_images_only(g_img, log_image, law_img, threshold_img, c_st, halftone_img):
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 4, 1)
    plt.imshow(g_img, cmap='gray')
    plt.title('Gray org Image')
    plt.axis('off')
    plt.subplot(3, 4, 2)
    plt.imshow(log_image, cmap='gray')
    plt.title('Log Image')
    plt.axis('off')
    plt.subplot(3, 4, 3)
    plt.imshow(law_img, cmap='gray')
    plt.title('Law Image')
    plt.axis('off')
    plt.subplot(3, 4, 6)
    plt.imshow(threshold_img, cmap='gray')
    plt.title('Thresholding Image')
    plt.axis('off')
    plt.subplot(3, 4, 7)
    plt.imshow(c_st, cmap='gray')
    plt.title('Contrast Stretched Image')
    plt.axis('off')
    plt.subplot(3, 4, 8)
    plt.imshow(halftone_img, cmap='gray')
    plt.title('Half-toned Image')
    plt.axis('off')
    plt.show()

"""**Display utility functions**"""

img_path = "image.png"
Log_image = Log_tranform(np.array(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)), 1)
power_law_img = power_law_T(np.array(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)), 1.2)
contrast_img = contrast_st(np.array(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)))
histrogram = calculate_histrogramgram(
np.array(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)))
histrogram_eq = histrogramgram_equalization(
np.array(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)))
t_image = thresolding(
np.array(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)), 120)
halftone_img = halftone(
np.array(cv2.imread(img_path, cv2.IMREAD_COLOR)))
org_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

display_images_only(
org_img,
Log_image,
power_law_img,
t_image,
contrast_img,
halftone_img)
display_histograms(
histrogram,
histrogram_eq
)