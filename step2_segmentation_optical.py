import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def step2_segmentation(original_img, gradient_img, radius=250, cx=None, cy=None):
    print("--- Step 2: Circular ROI-based Watershed Segmentation ---")

    # ---------------------------------------------------------
    # 1. CREATE A CIRCULAR REGION OF INTEREST (ROI)
    # ---------------------------------------------------------
    h, w = gradient_img.shape
    
    # If no center is provided, use the exact middle of the image
    if cx is None or cy is None:
        cx, cy = w // 2, h // 2
        
    r = int(radius) 
    
    # Create a black mask and draw a white circle in the middle
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    
    # Keep only the area inside the circle, make everything else black
    roi_gradient = cv2.bitwise_and(gradient_img, gradient_img, mask=mask)
    roi_original = cv2.bitwise_and(original_img, original_img, mask=mask)

    # ---------------------------------------------------------
    # 2. THRESHOLDING (Convert to Black and White)
    # ---------------------------------------------------------
    # Convert pixels > 20 to white (255), others to black (0)
    _, binary_final = cv2.threshold(roi_gradient, 20, 255, cv2.THRESH_BINARY)
    
    # ---------------------------------------------------------
    # 3. WATERSHED PREPARATION (Remove noise and find sure areas)
    # ---------------------------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Remove small noise dots (Morphological Opening)
    binary_final = cv2.morphologyEx(binary_final, cv2.MORPH_OPEN, kernel)
    
    # Find the "Sure Background" by expanding the white areas
    sure_bg = cv2.dilate(binary_final, kernel, iterations=3)
    
    # Find the "Sure Foreground" (center of particles) using Distance Transform
    dist = cv2.distanceTransform(binary_final, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.2 * dist.max(), 255, 0)
    
    # Find the "Unknown region" (boundaries we need to find)
    unknown = cv2.subtract(sure_bg, np.uint8(sure_fg))
    
    # ---------------------------------------------------------
    # 4. CREATE MARKERS FOR WATERSHED
    # ---------------------------------------------------------
    # Label the sure foreground areas with numbers (1, 2, 3...)
    _, markers = cv2.connectedComponents(np.uint8(sure_fg))
    
    # Add 1 to all labels so the background becomes 1 (instead of 0)
    markers = markers + 1
    
    # Mark the unknown boundary areas as 0
    markers[unknown == 255] = 0
    
    # ---------------------------------------------------------
    # 5. APPLY WATERSHED ALGORITHM
    # ---------------------------------------------------------
    img_color = cv2.cvtColor(roi_original, cv2.COLOR_BGR2RGB)
    
    # The watershed algorithm will fill the unknown areas and mark boundaries as -1
    markers = cv2.watershed(img_color, markers)
    
    # ---------------------------------------------------------
    # 6. FILTER BY SIZE AND PREPARE VISUALS
    # ---------------------------------------------------------
    final_markers = np.zeros_like(gradient_img, dtype=np.int32)
    particle_count = 0
    
    # Create empty images to draw the results
    display_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    watershed_output = np.zeros_like(display_img)
    
    # Loop through each detected particle
    for label in np.unique(markers):
        if label <= 1: continue # Skip background (1) and boundaries (-1)
        
        # Create a temporary mask for the current particle
        mask_label = np.where(markers == label, 1, 0).astype(np.uint8)
        area = np.sum(mask_label)
        
        # Only keep particles larger than 100 pixels
        if area > 100:
            final_markers[markers == label] = particle_count + 2
            particle_count += 1
            
            # Fill the detected particles with green color for visualization
            watershed_output[markers == label] = [0, 255, 0]
            display_img[markers == label] = [0, 255, 0]

    # Draw the particle boundaries in red color (-1)
    watershed_output[markers == -1] = [255, 0, 0]
    
    # Keep only the results inside the circular ROI
    watershed_output = cv2.bitwise_and(watershed_output, watershed_output, mask=mask)

    # ---------------------------------------------------------
    # 7. PLOT THE 4 STAGES FOR STREAMLIT UI
    # ---------------------------------------------------------
    # Create a layout with 4 side-by-side images
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor('#0E1117') # Match Streamlit dark background
    
    # Plot Stage 1: Gradient with ROI
    ax[0].imshow(roi_gradient, cmap='gray')
    ax[0].set_title("1. ROI Masked Gradient", color='white', fontsize=14)
    ax[0].axis("off")
    
    # Plot Stage 2: Binary Mask
    ax[1].imshow(binary_final, cmap='gray')
    ax[1].set_title("2. Binary Mask", color='white', fontsize=14)
    ax[1].axis("off")
    
    # Plot Stage 3: Watershed Result (Green fill, Red borders)
    ax[2].imshow(watershed_output)
    ax[2].set_title("3. Watershed Output", color='white', fontsize=14)
    ax[2].axis("off")
    
    # Plot Stage 4: Final Overlay on Original Image
    ax[3].imshow(display_img)
    ax[3].set_title(f"4. Detected Particles ({particle_count})", color='white', fontsize=14)
    ax[3].axis("off")

    plt.tight_layout()

    return final_markers, fig