import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def step2_segmentation(original_img, gradient_img, radius=250, cx=None, cy=None):
    print("--- Step 2: Circular ROI-based Watershed Segmentation ---")

    # 1. CIRCULAR ROI MASKING
    h, w = gradient_img.shape
    
    # Calculate the center point if cx and cy are not provided
    if cx is None or cy is None:
        cx, cy = w // 2, h // 2
        
    r = int(radius) 
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    
    # Mask the area outside the circle to black
    roi_gradient = cv2.bitwise_and(gradient_img, gradient_img, mask=mask)
    roi_original = cv2.bitwise_and(original_img, original_img, mask=mask)

    # 2. Thresholding
    _, binary_final = cv2.threshold(roi_gradient, 20, 255, cv2.THRESH_BINARY)
    
    # 3. Watershed Pre-processing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_final = cv2.morphologyEx(binary_final, cv2.MORPH_OPEN, kernel)
    
    dist = cv2.distanceTransform(binary_final, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.2 * dist.max(), 255, 0)
    
    sure_bg = cv2.dilate(binary_final, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, np.uint8(sure_fg))
    
    # 4. Marker Generation
    _, markers = cv2.connectedComponents(np.uint8(sure_fg))
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # 5. Apply Watershed
    img_color = cv2.cvtColor(roi_original, cv2.COLOR_BGR2RGB)
    markers = cv2.watershed(img_color, markers)
    
    # 6. Area-Based Filtering & Visualization
    final_markers = np.zeros_like(gradient_img, dtype=np.int32)
    particle_count = 0
    
    display_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    for label in np.unique(markers):
        if label <= 1: continue 
        
        mask_label = np.where(markers == label, 1, 0).astype(np.uint8)
        area = np.sum(mask_label)
        
        if area > 100:
            final_markers[markers == label] = particle_count + 2
            particle_count += 1
            # Highlight detected particle boundaries.
            display_img[markers == label] = [0, 255, 0]

    # 7. Prepare a figure for the UI
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(roi_gradient, cmap='gray'), ax[0].set_title("Circular ROI Gradient"), ax[0].axis("off")
    ax[1].imshow(display_img), ax[1].set_title(f"Detected Particles ({particle_count})"), ax[1].axis("off")

    return final_markers, fig
