import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Segmentation for the Nile Red Pipeline
def step2_segmentation(original_img, tophat_img):
    print("--- Step 2: Advanced Segmentation & Watershed Separation ---")

    # 1. Adaptive Thresholding
    binary_mask = cv2.adaptiveThreshold(
        tophat_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -5
    )

    # Morphological Opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 2. Sure Background
    sure_bg = cv2.dilate(clean_mask, kernel, iterations=3)

    # 3. Distance Transform
    dist_transform = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 4. Unknown Region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 5. Marker Labeling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 6. Apply Watershed
    img_color = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    watershed_img = img_color.copy()
    markers = cv2.watershed(watershed_img, markers)
    
    # Boundary labeling (Green)
    watershed_img[markers == -1] = [0, 255, 0] 

    # 7. Prepare a figure for the UI
    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    ax[0].imshow(clean_mask, cmap='gray'), ax[0].set_title("1. Binary Mask"), ax[0].axis("off")
    ax[1].imshow(cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX), cmap='viridis'), ax[1].set_title("2. Distance Transform"), ax[1].axis("off")
    ax[2].imshow(sure_fg, cmap='gray'), ax[2].set_title("3. Sure Foreground"), ax[2].axis("off")
    ax[3].imshow(watershed_img), ax[3].set_title("4. Watershed Cut"), ax[3].axis("off")
    
    return markers, fig
