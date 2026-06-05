import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st # Streamlit UI support

def step1_preprocessing(img_bgr): # Accept a BGR numpy image.
    print("--- Step 1: Advanced Pre-processing & FFT Frequency Analysis ---")

    # 1. Check if image is valid
    if img_bgr is None:
        st.error("Error: Image not found!")
        return None

    # Convert to RGB for displaying correctly
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. Color Space Conversion (RGB to HSV)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v_channel = cv2.split(img_hsv)

    # 3. FAST FOURIER TRANSFORM (FFT)
    dft = cv2.dft(np.float32(v_channel), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)

    rows, cols = v_channel.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    
    r = 150 
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - crow)**2 + (y - ccol)**2 <= r*r
    mask[mask_area] = 1

    fshift_filtered = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = cv2.idft(f_ishift)
    fft_filtered_img = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    
    fft_filtered_img = cv2.normalize(fft_filtered_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 4. Bilateral Filter
    filtered_v = cv2.bilateralFilter(fft_filtered_img, d=9, sigmaColor=75, sigmaSpace=75)

    # 5. Top-Hat
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    tophat_img = cv2.morphologyEx(filtered_v, cv2.MORPH_TOPHAT, kernel)

    # 6. Prepare a figure for the UI
    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    ax[0].imshow(img_rgb), ax[0].set_title("Original"), ax[0].axis("off")
    ax[1].imshow(v_channel, cmap='gray'), ax[1].set_title("HSV: V"), ax[1].axis("off")
    ax[2].imshow(magnitude_spectrum, cmap='gray'), ax[2].set_title("FFT Spectrum"), ax[2].axis("off")
    ax[3].imshow(fft_filtered_img, cmap='gray'), ax[3].set_title("FFT Filtered"), ax[3].axis("off")
    ax[4].imshow(tophat_img, cmap='gray'), ax[4].set_title("Top-Hat"), ax[4].axis("off")
    
    # Return the processed image and the UI figure.
    return tophat_img, fig
