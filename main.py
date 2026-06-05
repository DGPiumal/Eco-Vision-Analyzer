import cv2
from step3_feature_extraction import step3_feature_extraction

def show_menu():
    print("\n==================================================")
    print("        ECO-VISION ADVANCED PIPELINE MENU         ")
    print("==================================================")
    print("Select the type of image you are processing:")
    print("  [1] Nile Red / Clean Fluorescence Image")
    print("  [2] Optical Microscope Image (Gradient Method)")
    print("==================================================")
    return input("Enter your choice (1 or 2): ")

if __name__ == "__main__":
    user_choice = show_menu()
    image_path = 'Data-Set-2/b_2.jpg'
    #image_path = 'Data_Set/019.tiff'
    original_img = cv2.imread(image_path)

    if original_img is None:
        print("\n[ERROR] Image not found!")

    # --- PIPELINE SELECTION ---
    if user_choice == '1':
        print("\n[*] Loading Standard Pipeline (Nile Red)...")
        from step1_preprocessing import step1_preprocessing as preprocess
        from step2_segmentation import step2_segmentation as segment
        tophat_result = preprocess(image_path) # Use the standard preprocessing output
        markers, final_mask = segment(original_img, tophat_result)
        
    elif user_choice == '2':
        print("\n[*] Loading Optical Pi2peline (FFT + Gradient)...")
        # Import the optical preprocessing and segmentation modules
        from step1_preprocessing_optical import step1_preprocessing_optical as preprocess
        from step2_segmentation_optical import step2_segmentation as segment
        # Optical preprocessing returns a gradient image
        gradient_result = preprocess(image_path) 
        markers, final_mask = segment(original_img, gradient_result)
        
    else:
        print("[ERROR] Invalid choice.")
        exit()

    # --- STEP 3 (Common for both) ---
    final_result_img, statistics_df = step3_feature_extraction(original_img, markers)
    
    print("\n=== Eco-Vision Pipeline Completed Successfully! ===")
    print(statistics_df.head())
