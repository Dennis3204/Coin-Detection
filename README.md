# Coin Detector Using Classical Image Processing

## Overview

This project implements a lightweight, interactive coin detection and measurement system using **Python** and **OpenCV**, with a focus on simplicity, reproducibility, and zero reliance on machine learning or external datasets. The system processes images of coins to identify, label, and measure each coin using classical image processing techniques.

Developed as a final project for *CPE 462*, it demonstrates the power and limitations of threshold-based segmentation and contour analysis under controlled vs. real-world conditions.

---

## Features

- Grayscale conversion and Gaussian blur preprocessing
- Adaptive thresholding with Otsu’s method
- Morphological operations for noise reduction
- Contour detection and minimum enclosing circle fitting
- Overlap filtering to prevent redundant detections
- Interactive GUI using OpenCV:
  - Click to select a coin and view diameter
  - Displays measurements in pixels or mm (if scale is provided)
  - Navigate images with `'n'` (next) and `'q'` (quit)

---

## System Requirements

- Python 3.7 or newer
- OS: Windows, macOS, or Linux

### Python Dependencies

Install with:

```bash
pip install opencv-python numpy
