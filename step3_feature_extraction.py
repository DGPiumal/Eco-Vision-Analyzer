import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

# Preserve the existing Fractal Dimension function
def get_fractal_dimension(contour, img_shape):
    edge_image = np.zeros(img_shape, dtype=np.uint8)
    cv2.drawContours(edge_image, [contour], -1, 255, 1)
    x, y, w, h = cv2.boundingRect(contour)
    particle_edge = edge_image[y:y+h, x:x+w]
    max_box_size = min(w, h) // 2
    if max_box_size < 4: return 1.00
    box_sizes, box_counts = [], []
    for size in range(2, max_box_size, 2):
        count = 0
        for row in range(0, h, size):
            for col in range(0, w, size):
                box = particle_edge[row:row+size, col:col+size]
                if np.any(box > 0): count += 1
        if count > 0:
            box_sizes.append(size)
            box_counts.append(count)
    if len(box_sizes) < 2: return 1.00
    x_log = np.log(box_sizes)
    y_log = np.log(box_counts)
    slope, intercept = np.polyfit(x_log, y_log, 1)
    return abs(slope)

def step3_feature_extraction(original_img, markers):
    print("--- Step 3: Morphometrics, Texture & Fractal Dimension ---")

    particle_mask = np.zeros_like(markers, dtype=np.uint8)
    particle_mask[markers > 1] = 255
    contours, _ = cv2.findContours(particle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    gray_original = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    img_shape = gray_original.shape
    particle_data = []
    result_img = original_img.copy()

    particle_id = 1
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10: continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0: continue
        
        circularity = (4 * math.pi * area) / (perimeter * perimeter)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        roi_gray = gray_original[y:y+h, x:x+w]
        glcm = graycomatrix(roi_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        fd = get_fractal_dimension(contour, img_shape)

        if aspect_ratio > 3.0 or circularity < 0.3:
            p_type, color = "Fiber", (255, 0, 0)
        elif circularity > 0.8 and homogeneity > 0.4:
            p_type, color = "Pellet", (0, 255, 255)
        else:
            p_type, color = "Fragment", (0, 255, 0)

        particle_data.append({"ID": particle_id, "Type": p_type, "Area": round(area, 2), "FD": round(fd, 3)})
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result_img, f"#{particle_id} {p_type}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        particle_id += 1

    df = pd.DataFrame(particle_data)
    
    # Dashboard Figure
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f"Eco-Vision Output ({len(df)} Particles)")
    ax[0].axis("off")
    
    if not df.empty:
        counts = df['Type'].value_counts()
        counts.plot(kind='bar', color=['blue', 'green', 'yellow'], ax=ax[1])
        ax[1].set_title("Classification Statistics")
        ax[1].tick_params(axis='x', rotation=45)
        ax[1].set_xlabel("Particle Type")
        ax[1].set_ylabel("Count")

    plt.tight_layout()
    return result_img, df, fig