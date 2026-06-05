import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def step1_preprocessing_optical(img_bgr):
    print("--- Step 1: Optimized Optical Pre-processing ---")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. FFT Denoising
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    cv2.circle(mask, (ccol, crow), 80, (1, 1), -1)
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    fft_img = cv2.idft(f_ishift)
    fft_img = cv2.magnitude(fft_img[:,:,0], fft_img[:,:,1])
    fft_img = cv2.normalize(fft_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 2. Inversion
    inverted = cv2.bitwise_not(fft_img)

    # 3. Morphological Gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gradient = cv2.morphologyEx(inverted, cv2.MORPH_GRADIENT, kernel)

    # 4. Final Contrast Stretching
    final = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)

    # 5. Prepare figure for the UI
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    ax[0].imshow(gray, cmap='gray'), ax[0].set_title("1. Original"), ax[0].axis("off")
    ax[1].imshow(fft_img, cmap='gray'), ax[1].set_title("2. FFT"), ax[1].axis("off")
    ax[2].imshow(gradient, cmap='gray'), ax[2].set_title("3. Gradient"), ax[2].axis("off")
    ax[3].imshow(final, cmap='gray'), ax[3].set_title("4. Final Processed"), ax[3].axis("off")
    
    return final, fig