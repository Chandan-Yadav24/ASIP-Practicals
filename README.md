# ASIP Practical Programs - MU MSC CS Semester 1 (2025-26)
## Advanced Signal and Image Processing Practicals - Mumbai University

**ASIP Practicals | MSC CS Mumbai University | Semester 1 | Signal Processing | Image Processing | All 11 Practicals**

This repository contains **all 11 complete practical implementations** for the **ASIP (Advanced Signal and Image Processing)** subject as per the **Mumbai University MSC CS curriculum for Semester 1, 2025-26 academic year**. These practicals cover comprehensive signal processing and image processing techniques including FFT, convolution, edge detection, segmentation, and morphological operations.

### Keywords:
ASIP practicals, MU practicals, MSC CS practicals, Mumbai University signal processing, image processing practical, semester 1 practicals, advanced signal processing, digital image processing, computer vision practicals, Python signal processing

---

## Overview of ASIP Course

The **Advanced Signal and Image Processing (ASIP)** course is a core subject in the **MSC Computer Science** program at **Mumbai University**. This Semester 1 practical course introduces students to advanced techniques in signal and image processing, including:

- **Digital Signal Processing:** FFT, upsampling, downsampling, convolution
- **Image Enhancement:** Intensity transformations, histogram equalization, filtering
- **Edge Detection:** Sobel operator, Canny edge detection algorithm
- **Morphological Operations:** Image erosion, dilation, opening, closing
- **Image Segmentation:** Threshold-based, region-based, edge-based segmentation methods
- **Feature Extraction:** Corner detection, blob detection, HoG features, Haar features
- **Pattern Recognition:** Template matching, shape detection, circle and line detection

All practicals are implemented in **Python** using popular libraries like **OpenCV, NumPy, SciPy, and Scikit-image**.

---

## Complete List of 11 ASIP Practicals - MSC CS Semester 1

### Practical 1: Signal Processing Fundamentals - Upsampling, Downsampling, FFT
**File:** `practical_no_1.py`
**Topics:** Signal processing, upsampling, downsampling, DFT, Fast Fourier Transform, frequency domain analysis

Demonstrates the following aspects of signal processing on suitable data:
- Upsampling and downsampling on Image/speech signal
- Fast Fourier Transform (FFT) to compute DFT
- Frequency domain signal analysis
- Sample rate conversion techniques

---

### Practical 2: Convolution and Template Matching - Signal Processing
**File:** `practical_no_2.py`
**Topics:** Convolution operation, template matching, correlation, 2D filtering, image matching

Demonstrates the following aspects on sound/image data:
- Convolution operation and its applications
- Template Matching techniques for pattern recognition
- Spatial filtering in image processing
- Correlation-based image analysis

---

### Practical 3: Point/Pixel Intensity Transformations - Image Processing
**File:** `practical_no_3.py`
**Topics:** Intensity transformations, logarithmic transformation, power-law transformation, contrast enhancement, histogram equalization, thresholding, halftoning

Implements point/pixel intensity transformations such as:
- Log and Power-law transformations for image enhancement
- Contrast adjustments and contrast stretching
- Histogram equalization for brightness normalization
- Thresholding and halftoning operations for binary image processing

---

### Practical 4: Image Enhancement using Derivatives - Gradient and Laplacian
**File:** `practical_no_4.py`
**Topics:** Image derivatives, gradient operations, Laplacian operator, edge enhancement, sharpening

Applies various enhancements on images using image derivatives by implementing:
- Gradient operations (Sobel, Prewitt gradients)
- Laplacian operations for edge detection
- Image derivative-based enhancement techniques
- First and second derivative operators

---

### Practical 5: Linear and Nonlinear Noise Smoothing - Image/Signal Denoising
**File:** `practical_no_5.py`
**Topics:** Noise smoothing, filtering, Gaussian blur, median filter, bilateral filter, noise reduction

Implements linear and nonlinear noise smoothing on suitable image or sound signal:
- Linear filtering techniques (Gaussian blur, box filter)
- Nonlinear filtering (median filter, morphological filters)
- Bilateral filtering for edge-preserving smoothing
- Noise removal and signal denoising

---

### Practical 6: Image Enhancement Filters - Smoothing, Sharpening, Unsharp Masking
**File:** `practical_no_6.py`
**Topics:** Image filtering, smoothing filters, sharpening filters, unsharp masking, image enhancement techniques

Applies various image enhancement using image derivatives by implementing:
- Smoothing filters for noise reduction
- Sharpening filters for edge enhancement
- Unsharp masking for selective sharpening
- Filter design for specific application requirements

---

### Practical 7: Edge Detection Techniques - Sobel and Canny Algorithms
**File:** `practical_no_7.py`
**Topics:** Edge detection, Sobel edge detection, Canny edge detection, edge extraction, boundary detection, feature extraction

Applies edge detection techniques such as:
- Sobel edge detector for gradient-based edge detection
- Canny edge detection algorithm for optimal edge detection
- Multi-stage edge detection pipeline
- Feature extraction from edge maps for meaningful image analysis

---

### Practical 8: Morphological Image Processing Operations
**File:** `practical_no_8.py`
**Topics:** Morphological operations, erosion, dilation, opening, closing, morphological transformations

Implements various morphological image processing techniques:
- Erosion operation for shrinking objects
- Dilation operation for expanding objects
- Opening operation (erosion followed by dilation)
- Closing operation (dilation followed by erosion)
- Advanced morphological transformations

---

### Practical 9: Image Feature Extraction - Corner Detection, Blob Detection, HoG, Haar Features
**File:** `practical_no_9.py`
**Topics:** Feature extraction, corner detection, blob detection, HoG features, Haar features, image descriptors

Extracts image features by implementing methods like:
- Corner detectors (Harris corner detection)
- Blob detectors for region-of-interest identification
- Histogram of Oriented Gradients (HoG) feature descriptor
- Haar features for cascade classification
- Feature-based image analysis and recognition

---

### Practical 10: Image Segmentation - Lines, Circles, Shapes Detection (Part 1)
**File:** `practical_no_10.py`
**Topics:** Image segmentation, Hough transform, line detection, circle detection, shape detection, edge-based segmentation, region-based segmentation

Applies segmentation for detecting:
- Lines using Hough Line Transform
- Circles using Hough Circle Transform
- Other shapes and objects detection
- Edge-based segmentation methods
- Region-based segmentation techniques

---

### Practical 11: Advanced Image Segmentation - Shapes and Objects Detection (Part 2)
**File:** `practical_no_11.py`
**Topics:** Advanced segmentation, contour detection, shape analysis, object detection, instance segmentation, semantic segmentation

Applies advanced segmentation for detecting:
- Lines, circles, and complex shapes
- Multiple objects and instances in image
- Advanced edge-based segmentation algorithms
- Advanced region-based segmentation methods
- Contour analysis and shape properties extraction

---

## Requirements

All required dependencies are listed in `requirements.txt`. To install them, run:

```bash
pip install -r requirements.txt
```

Common packages used in these ASIP practicals include:
- **NumPy** - Numerical computing and array operations
- **OpenCV (cv2)** - Computer Vision and image processing library
- **Matplotlib** - Data visualization and image display
- **SciPy** - Scientific computing and signal processing
- **Scikit-image** - Advanced image processing algorithms
- **Pillow** - Python Imaging Library for image manipulation

---

## Setup Instructions

### 1. Create Python Virtual Environment

**Windows:**
```bash
python -m venv venv
```

**Linux/Mac:**
```bash
python3 -m venv venv
```

### 2. Activate Virtual Environment

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```bash
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Install All Required Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- numpy
- opencv-python
- matplotlib
- scipy
- scikit-image
- pillow

### 4. Verify Installation

```bash
python -c "import cv2; import numpy; import scipy; print('All packages installed successfully!')"
```

---

## Running the ASIP Practicals

Execute any practical using Python:

**Run individual practicals:**
```bash
python practical_no_1.py
python practical_no_2.py
python practical_no_3.py
# ... continue for practical_no_4 through practical_no_11
```

**Run from PowerShell (if using venv):**
```bash
& "venv\Scripts\Activate.ps1"
python practical_no_1.py
```

**Run all practicals sequentially:**
```bash
for %i in (1,2,3,4,5,6,7,8,9,10,11) do python practical_no_%i.py
```

---

## Important Notes & Guidelines

### File Paths & Media
- Ensure all input images/audio files are in supported formats (JPG, PNG, BMP, WAV, MP3, etc.)
- Modify file paths in scripts as needed based on your directory structure
- Create a `/data` or `/images` folder for storing input files

### Sample Data
- Some practicals may require sample datasets
- Ensure these are available before running the practicals
- Common sources: OpenCV sample images, public datasets, your own data

### Troubleshooting
- **Import Error:** Ensure all packages are installed with `pip install -r requirements.txt`
- **File Not Found:** Check image/audio file paths in the script
- **OpenCV Issues:** Make sure OpenCV is correctly installed: `pip install --upgrade opencv-python`

---

## Topics Covered - Comprehensive ASIP Curriculum

### Signal Processing Topics:
- **Digital Signal Processing (DSP):** Fast Fourier Transform (FFT), Discrete Fourier Transform (DFT)
- **Signal Sampling:** Upsampling, downsampling, sample rate conversion
- **Convolution:** Linear convolution, circular convolution, 2D convolution
- **Frequency Domain Analysis:** Spectrum analysis, frequency response

### Image Processing & Enhancement Topics:
- **Intensity Transformations:** Logarithmic transformation, power-law transformation, gamma correction
- **Histogram Processing:** Histogram equalization, contrast stretching, histogram matching
- **Spatial Filtering:** Box filter, Gaussian blur, median filter, bilateral filter
- **Image Derivatives:** Gradient operators, Laplacian operator, directional derivatives
- **Sharpening Techniques:** Unsharp masking, high-pass filtering, edge enhancement

### Edge Detection & Analysis:
- **Edge Detection Algorithms:** Sobel operator, Canny edge detector, Roberts operator
- **Edge Properties:** Edge thinning, edge linking, boundary extraction
- **Feature Extraction:** Corner detection (Harris), blob detection, feature descriptors
- **Shape Detection:** Hough transform, line detection, circle detection

### Image Segmentation:
- **Thresholding Methods:** Global thresholding, adaptive thresholding, Otsu's method
- **Region-based Segmentation:** Region growing, watershed segmentation, k-means clustering
- **Edge-based Segmentation:** Active contours, boundary detection, contour tracing
- **Object Detection:** Template matching, pattern recognition, shape matching

### Morphological Operations:
- **Basic Operations:** Erosion, dilation, opening, closing
- **Advanced Operations:** Skeletonization, thinning, hit-or-miss transform
- **Morphological Analysis:** Convex hull, distance transform

### Feature Descriptors & Detection:
- **HoG (Histogram of Oriented Gradients):** Feature extraction for object detection
- **Haar Features:** Cascade classifiers, Haar-like features
- **Scale-Invariant Features:** Corner detection, blob detection
- **Keypoint Detection:** Interest point detection and matching

---

## University Information

**Institution:** Mumbai University (MU)  
**Program:** Master of Science in Computer Science (MSC CS)  
**Subject:** Advanced Signal and Image Processing (ASIP)  
**Course Code:** ASIP (Practical)  
**Semester:** 1 (First Semester)  
**Academic Year:** 2025-26  
**Batch:** 2025-2027  
**Total Practicals:** 11

---

## Who Should Use This Repository

This repository is ideal for:
- **MSC CS Students:** All students enrolled in MSC Computer Science at Mumbai University
- **ASIP Course Students:** Students taking Advanced Signal and Image Processing practical course
- **Image Processing Learners:** Anyone learning digital image processing and signal processing
- **Computer Vision Enthusiasts:** Individuals interested in computer vision applications
- **Signal Processing Practitioners:** Professionals working with signal and image analysis

---

## Key Features

✓ **Complete Coverage:** All 11 practicals of ASIP course  
✓ **Practical Implementation:** Working Python code for each topic  
✓ **Semester 1 Aligned:** Follows Mumbai University MSC CS Semester 1 curriculum  
✓ **Well-Documented:** Detailed descriptions for each practical  
✓ **Easy Setup:** Simple installation and execution instructions  
✓ **Real-World Applications:** Practical examples applicable in industry  
✓ **Multiple Algorithms:** Implementation of multiple techniques for each topic  

---

## How to Use This Repository

1. **Clone or Download** the repository
2. **Follow setup instructions** below
3. **Run individual practicals** as per your course schedule
4. **Modify and experiment** with the code for learning
5. **Reference** while working on assignments and projects

---

## Common Search Terms & Related Topics

**Search Keywords:** 
- ASIP practicals MU
- MSC CS practical assignments
- Mumbai University signal processing
- Advanced signal and image processing
- Digital image processing practical
- Computer vision practical programs
- Image processing with Python OpenCV
- Signal processing with NumPy SciPy
- Edge detection algorithms
- Image segmentation techniques
- Morphological operations image processing
- Feature extraction computer vision

---

## Related Learning Resources

- **OpenCV Documentation:** https://docs.opencv.org/
- **NumPy Tutorials:** https://numpy.org/doc/
- **SciPy Documentation:** https://docs.scipy.org/
- **Scikit-Image:** https://scikit-image.org/
- **Digital Image Processing:** Gonzalez & Woods reference
- **Signal Processing:** Oppenheim & Schafer reference

---

## Future Enhancements

- [ ] Add detailed comments in each practical code
- [ ] Include sample test images and audio files
- [ ] Create video tutorials for each practical
- [ ] Add Jupyter notebook versions of each practical
- [ ] Include performance benchmarking
- [ ] Add real-world application examples

---

## Contribution Guidelines

If you have improvements or additional examples:
1. Test your code thoroughly
2. Follow PEP 8 Python style guidelines
3. Add comments and documentation
4. Submit with clear descriptions

---

## Disclaimer & License

**For Academic Purposes Only**

This repository contains practical code implementations for the Advanced Signal and Image Processing (ASIP) subject as per Mumbai University's MSC CS curriculum. 

**Usage:**
- Use as a reference for learning signal and image processing concepts
- Modify and adapt code for your specific requirements
- Do NOT directly copy-paste for assignment submission without understanding
- Follow your institution's academic integrity and plagiarism policies

**License:** For educational use within Mumbai University and authorized institutions only.

---

## Support & Questions

For questions regarding:
- **Course Content:** Contact your ASIP course instructor
- **Code Issues:** Review the practical implementations and test with different inputs
- **Environment Setup:** Refer to the setup instructions above or consult Python/pip documentation

---

**Last Verified:** February 2026  
**Python Version:** 3.8 or higher  
**OpenCV Version:** 4.5 or higher  
**NumPy Version:** 1.21 or higher
