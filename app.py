import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates 

# Pipeline Imports
from step1_preprocessing_optical import step1_preprocessing_optical as preprocess_opt
from step2_segmentation_optical import step2_segmentation as segment_opt
from step1_preprocessing import step1_preprocessing as preprocess_std
from step2_segmentation import step2_segmentation as segment_std
from step3_feature_extraction import step3_feature_extraction

st.set_page_config(page_title="Eco-Vision Analyzer", layout="wide")
st.title("🔬 Eco-Vision Advanced Particle Analyzer")

# App Logo
try:
    st.sidebar.image("logo.svg", use_container_width=True)
except FileNotFoundError:
    pass

# 1. Sidebar - Settings
pipeline = st.sidebar.selectbox("Choose Pipeline", ["Nile Red", "Optical (FFT + Gradient)"])

status_placeholder = st.sidebar.empty()
status_placeholder.info("Ready to process...")

# 2. ROI controls
if 'cx' not in st.session_state: st.session_state.cx = 500
if 'cy' not in st.session_state: st.session_state.cy = 500
if 'last_click' not in st.session_state: st.session_state.last_click = None

r_radius = 250 
if pipeline == "Optical (FFT + Gradient)":
    st.sidebar.subheader("ROI Controls")
    r_radius = st.sidebar.slider("Radius", 50, 500, 250)
    
    # Use session state values as the slider defaults.
    new_cx = st.sidebar.slider("Center X", 0, 1000, value=int(st.session_state.cx))
    new_cy = st.sidebar.slider("Center Y", 0, 1000, value=int(st.session_state.cy))
    
    # Store slider values in session state.
    st.session_state.cx = new_cx
    st.session_state.cy = new_cy

uploaded_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'png', 'tiff'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    
    if pipeline == "Optical (FFT + Gradient)":
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("ROI Preview (Click to re-center)")
            value = streamlit_image_coordinates(
                cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), 
                width=400,
                key="img_clicker"
            )
            
            # Recenter the ROI after a new image click.
            if value is not None and value != st.session_state.last_click:
                st.session_state.cx = value["x"]
                st.session_state.cy = value["y"]
                st.session_state.last_click = value
                st.rerun() 
        
        with col_right:
            st.subheader("Adjust ROI")
            preview = original_img.copy()
            cv2.circle(preview, (st.session_state.cx, st.session_state.cy), r_radius, (0, 255, 0), 5)
            st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), width=400, caption="Live ROI View")
    else:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), width=400, caption="Original")

    if st.button("🚀 Run Analysis"):
        try:
            status_placeholder.write("🔄 Step 1: Pre-processing...")
            if pipeline == "Nile Red":
                tophat_result, fig1 = preprocess_std(original_img)
            else:
                gradient_result, fig1 = preprocess_opt(original_img)
            
            status_placeholder.write("🔄 Step 2: Segmentation...")
            if pipeline == "Nile Red":
                markers, fig2 = segment_std(original_img, tophat_result)
            else:
                markers, fig2 = segment_opt(original_img, gradient_result, radius=r_radius, cx=st.session_state.cx, cy=st.session_state.cy)
            
            status_placeholder.write("🔄 Step 3: Feature Extraction...")
            final_result_img, stats_df, fig3 = step3_feature_extraction(original_img, markers)
            
            status_placeholder.success("✅ Analysis Complete!")
            
            tab1, tab2, tab3 = st.tabs(["Pre-processing", "Detection", "Data Report"])
            
            with tab1: st.pyplot(fig1)
            with tab2: 
                col_orig, col_det = st.columns(2)
                with col_orig:
                    st.write("### Original (within ROI)")
                    st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col_det:
                    st.write("### Detected Particles")
                    st.image(cv2.cvtColor(final_result_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.pyplot(fig2)
            with tab3: 
                st.write("### Final Detection Visualization")
                st.image(cv2.cvtColor(final_result_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                col_data, col_graph = st.columns([1, 1])
                with col_data:
                    st.write("### Particle Statistics")
                    st.dataframe(stats_df)
                    # Download CSV
                    csv = stats_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download CSV Report", csv, "particle_analysis.csv", "text/csv")
                with col_graph:
                    st.write("### Classification Distribution")
                    if not stats_df.empty:
                        counts = stats_df['Type'].value_counts()
                        st.bar_chart(counts)
                st.write("### Advanced Analysis Dashboard")
                st.pyplot(fig3, use_container_width=True)
                
        except Exception as e:
            status_placeholder.error("❌ Analysis Failed!")
            st.error(f"Error: {e}")
