# -*- coding: utf-8 -*-
"""
Gradio Frontend for Enhanced Plant Disease Detection
Showcases DIP techniques and provides interactive interface
"""

import gradio as gr
import numpy as np
import cv2
import pickle
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import io
from PIL import Image
import sys

try:
    from skimage.feature import corner_harris, corner_peaks
except ImportError:
    print("⚠️  scikit-image not found. Please install: pip install scikit-image")
    sys.exit(1)

from enhanced_br import DIPPreprocessor, EnhancedModel, SmartEvaluator

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

BASE_PATH = Path.cwd()
MODEL_DIR = BASE_PATH / 'models'
CHECKPOINT_DIR = BASE_PATH / 'checkpoints'

dip_preprocessor = DIPPreprocessor()
enhanced_model = None
class_names = None
cnn_model = None

# ============================================================================
# LOAD MODELS
# ============================================================================

def load_models():
    """Load enhanced model components"""
    global enhanced_model, class_names, cnn_model
    
    try:
        # Load enhanced model metadata
        model_path = MODEL_DIR / 'enhanced_model.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Load CNN model
            cnn_model_path = Path(model_data['cnn_model_path'])
            if cnn_model_path.exists():
                cnn_model = keras.models.load_model(cnn_model_path)
                print(f"✅ Loaded CNN model: {cnn_model_path.name}")
            else:
                # Try final model
                final_model = CHECKPOINT_DIR / 'final_model.keras'
                if final_model.exists():
                    cnn_model = keras.models.load_model(final_model)
                    print(f"✅ Loaded final model")
                else:
                    print("⚠️  No CNN model found, using basic predictions")
                    cnn_model = None
            
            # Load RF model and scalers
            enhanced_model = EnhancedModel(
                len(model_data['class_names']),
                model_data['feature_dim']
            )
            enhanced_model.rf_model = model_data['rf_model']
            enhanced_model.scaler = model_data['scaler']
            enhanced_model.pca = model_data['pca']
            enhanced_model.cnn_model = cnn_model
            enhanced_model.dip_preprocessor = dip_preprocessor
            
            class_names = model_data['class_names']
            
            print(f"✅ Loaded enhanced model with {len(class_names)} classes")
            return True
        else:
            print("⚠️  Enhanced model not found. Please run enhanced_br.py first")
            return False
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_disease(image):
    """Predict plant disease from image"""
    if image is None:
        return "Please upload an image", None, None
    
    try:
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Get IMG_SIZE from enhanced_br
        from enhanced_br import IMG_SIZE
        
        # Resize if needed
        if image.shape[0] != IMG_SIZE or image.shape[1] != IMG_SIZE:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        
        # Get prediction
        if enhanced_model and cnn_model:
            # Ensemble prediction
            pred_proba = enhanced_model.predict_ensemble(image, cnn_model)
        elif cnn_model:
            # CNN only
            preprocessed = dip_preprocessor.preprocess_image(image)
            preprocessed_batch = np.expand_dims(preprocessed, axis=0)
            pred_proba = cnn_model.predict(preprocessed_batch, verbose=0)[0]
        else:
            # Basic prediction using features only
            # Use 224x224 for feature extraction to match training
            features, _ = dip_preprocessor.extract_all_features(image, feature_size=224)
            if enhanced_model and enhanced_model.rf_model:
                features_scaled = enhanced_model.scaler.transform([features])
                features_pca = enhanced_model.pca.transform(features_scaled)
                pred_proba = enhanced_model.rf_model.predict_proba(features_pca)[0]
            else:
                return "Model not loaded. Please run enhanced_br.py first", None, None
        
        # Get top predictions
        top_indices = np.argsort(pred_proba)[-5:][::-1]
        
        # Format results
        result_text = "🌿 PLANT DISEASE DETECTION RESULTS\n"
        result_text += "=" * 50 + "\n\n"
        
        for i, idx in enumerate(top_indices):
            confidence = pred_proba[idx] * 100
            disease_name = class_names[idx] if class_names else f"Class {idx}"
            result_text += f"{i+1}. {disease_name}\n"
            result_text += f"   Confidence: {confidence:.2f}%\n\n"
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Top predictions bar chart
        top_probs = [pred_proba[idx] * 100 for idx in top_indices]
        top_labels = [class_names[idx] if class_names else f"Class {idx}" 
                     for idx in top_indices]
        
        axes[1].barh(range(len(top_labels)), top_probs, color='green', alpha=0.7)
        axes[1].set_yticks(range(len(top_labels)))
        axes[1].set_yticklabels([label.replace('___', ' - ').replace('_', ' ') 
                                 for label in top_labels], fontsize=9)
        axes[1].set_xlabel('Confidence (%)', fontsize=10)
        axes[1].set_title('Top 5 Predictions', fontsize=12, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close()
        
        return result_text, result_image, pred_proba
        
    except Exception as e:
        return f"Error: {str(e)}", None, None

# ============================================================================
# DIP VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_dip_techniques(image):
    """Visualize all DIP preprocessing techniques"""
    if image is None:
        return None
    
    try:
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Get IMG_SIZE from enhanced_br
        from enhanced_br import IMG_SIZE
        
        # Resize
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        
        # Apply DIP techniques
        intensity = dip_preprocessor.intensity_transformation(image)
        histogram = dip_preprocessor.histogram_equalization(image)
        spatial = dip_preprocessor.spatial_filtering(image)
        morphological = dip_preprocessor.morphological_processing(image)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('1. Original Image', fontsize=11, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Intensity Transformation
        axes[0, 1].imshow(intensity)
        axes[0, 1].set_title('2. Intensity Transformation\n(Gamma Correction + Contrast)', 
                            fontsize=11, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Histogram Equalization
        axes[0, 2].imshow(histogram)
        axes[0, 2].set_title('3. Histogram Processing\n(CLAHE)', 
                            fontsize=11, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Spatial Filtering
        axes[1, 0].imshow(spatial)
        axes[1, 0].set_title('4. Spatial Filtering\n(Gaussian + Median + Unsharp)', 
                            fontsize=11, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Morphological Processing
        if len(morphological.shape) == 3:
            axes[1, 1].imshow(morphological[:, :, 0], cmap='gray')
        else:
            axes[1, 1].imshow(morphological, cmap='gray')
        axes[1, 1].set_title('5. Morphological Processing\n(Opening + Closing + Gradient)', 
                            fontsize=11, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Final Enhanced
        enhanced = dip_preprocessor.preprocess_image(image)
        axes[1, 2].imshow(enhanced)
        axes[1, 2].set_title('6. Final Enhanced Image\n(Combined DIP Pipeline)', 
                            fontsize=11, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.suptitle('Digital Image Processing (DIP) Techniques', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close()
        
        return result_image
        
    except Exception as e:
        return None

def visualize_features(image):
    """Visualize extracted features (HoG, SIFT, etc.)"""
    if image is None:
        return None
    
    try:
        # Convert and preprocess
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Get IMG_SIZE from enhanced_br
        from enhanced_br import IMG_SIZE
        
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        
        # Extract features - use 224x224 for consistent feature dimensions
        features, feature_dict = dip_preprocessor.extract_all_features(image, feature_size=224)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # HoG visualization
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hog_features, hog_image = feature.hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            visualize=True
        )
        axes[0, 0].imshow(hog_image, cmap='hot')
        axes[0, 0].set_title(f'HoG Features\n({len(feature_dict["hog"])} dimensions)', 
                            fontsize=11, fontweight='bold')
        axes[0, 0].axis('off')
        
        # SIFT keypoints
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sift = cv2.SIFT_create(nfeatures=50)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        img_sift = cv2.drawKeypoints(image, keypoints, None, 
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        axes[0, 1].imshow(img_sift)
        axes[0, 1].set_title(f'SIFT Features\n({len(keypoints)} keypoints)', 
                            fontsize=11, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Corner detection
        try:
            corners = corner_harris(gray, method='k', k=0.04, sigma=1)
            corner_coords = corner_peaks(corners, min_distance=5, threshold_abs=0.02)
        except:
            # Fallback to OpenCV
            corners_cv = cv2.cornerHarris(gray.astype(np.float32), 2, 3, 0.04)
            corner_coords = np.argwhere(corners_cv > 0.01 * corners_cv.max())
            if len(corner_coords) == 0:
                corner_coords = np.array([]).reshape(0, 2)
        img_corners = image.copy()
        for coord in corner_coords:
            cv2.circle(img_corners, (coord[1], coord[0]), 5, (255, 0, 0), -1)
        axes[1, 0].imshow(img_corners)
        axes[1, 0].set_title(f'Corner Detection\n({len(corner_coords)} corners)', 
                            fontsize=11, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Feature statistics
        feature_stats = {
            'HoG': len(feature_dict['hog']),
            'SIFT': len(feature_dict['sift']) // 128,
            'Moments': len(feature_dict['moments']),
            'Corners': int(feature_dict['corners'][0])
        }
        
        axes[1, 1].bar(feature_stats.keys(), feature_stats.values(), color=['green', 'blue', 'orange', 'red'])
        axes[1, 1].set_title('Feature Statistics', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Count/Dimensions')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Feature Extraction Techniques', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close()
        
        return result_image
        
    except Exception as e:
        print(f"Error in feature visualization: {e}")
        return None

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """Create Gradio interface"""
    
    # Load models
    models_loaded = load_models()
    
    with gr.Blocks(title="Enhanced Plant Disease Detection") as app:
        gr.Markdown("""
        # 🌿 Enhanced Plant Disease Detection System
        
        ## Digital Image Processing (DIP) Techniques Showcase
        
        This system demonstrates advanced image processing techniques:
        - **Intensity Transformation**: Gamma correction and contrast enhancement
        - **Histogram Processing**: Adaptive histogram equalization (CLAHE)
        - **Spatial Filtering**: Gaussian, median, and unsharp masking
        - **Morphological Processing**: Opening, closing, and gradient operations
        - **Feature Extraction**: HoG, SIFT, Moments, and Corner Detection
        - **Ensemble Learning**: CNN + Traditional CV features fusion
        
        Upload a plant leaf image to see the DIP techniques in action and get disease predictions!
        """)
        
        with gr.Tabs():
            with gr.Tab("🔬 Disease Detection"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(label="Upload Plant Leaf Image", type="numpy")
                        predict_btn = gr.Button("🔍 Detect Disease", variant="primary", size="lg")
                    
                    with gr.Column():
                        output_text = gr.Textbox(label="Prediction Results", lines=10)
                        output_chart = gr.Image(label="Visualization")
                
                predict_btn.click(
                    fn=predict_disease,
                    inputs=[input_image],
                    outputs=[output_text, output_chart, gr.State()]
                )
            
            with gr.Tab("🖼️ DIP Techniques"):
                with gr.Row():
                    with gr.Column():
                        dip_input = gr.Image(label="Upload Image for DIP Processing", type="numpy")
                        dip_btn = gr.Button("🔄 Apply DIP Techniques", variant="primary")
                    
                    with gr.Column():
                        dip_output = gr.Image(label="DIP Processing Results")
                
                dip_btn.click(
                    fn=visualize_dip_techniques,
                    inputs=[dip_input],
                    outputs=[dip_output]
                )
            
            with gr.Tab("🔍 Feature Extraction"):
                with gr.Row():
                    with gr.Column():
                        feat_input = gr.Image(label="Upload Image for Feature Extraction", type="numpy")
                        feat_btn = gr.Button("📊 Extract Features", variant="primary")
                    
                    with gr.Column():
                        feat_output = gr.Image(label="Feature Visualization")
                
                feat_btn.click(
                    fn=visualize_features,
                    inputs=[feat_input],
                    outputs=[feat_output]
                )
            
            with gr.Tab("📊 Model Information"):
                gr.Markdown("""
                ## Model Architecture
                
                ### 1. CNN Branch (MobileNetV2)
                - **Base Model**: MobileNetV2 (ImageNet pretrained)
                - **Input Size**: 108x108x3 (optimized for laptop)
                - **Fine-tuning**: Enabled
                - **Output**: 38-class classification
                
                ### 2. Traditional Features Branch
                - **HoG Features**: 324 dimensions
                - **SIFT Features**: 1280 dimensions
                - **Moments**: 10 dimensions
                - **Corner Features**: 3 dimensions
                - **Total**: ~1617 dimensions
                
                ### 3. Ensemble Fusion
                - **CNN Weight**: 70%
                - **RF Weight**: 30%
                - **Smart Boosting**: Applied for confidence enhancement
                
                ### 4. Performance Metrics
                - **Enhanced Accuracy**: 82-85% (with DIP preprocessing)
                - **Standard Accuracy**: Baseline from CNN
                - **F1-Score**: Weighted average across classes
                
                ## DIP Techniques Applied
                
                1. **Intensity Transformation**
                   - Gamma correction (γ = 0.8)
                   - Contrast stretching (2-98 percentile)
                
                2. **Histogram Processing**
                   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
                   - Clip limit: 2.0, Tile size: 8x8
                
                3. **Spatial Filtering**
                   - Gaussian blur (5x5, σ=1.0)
                   - Median filter (5x5)
                   - Unsharp masking (enhancement factor: 1.5)
                
                4. **Morphological Processing**
                   - Opening (erosion + dilation)
                   - Closing (dilation + erosion)
                   - Gradient (edge detection)
                
                5. **Feature Extraction**
                   - HoG: 9 orientations, 16x16 pixels/cell
                   - SIFT: 100 keypoints max
                   - Hu Moments: 7 invariant moments
                   - Harris Corner Detection
                """)
        
        gr.Markdown("""
        ---
        **Note**: This system uses advanced DIP techniques to enhance image quality and extract 
        robust features for improved disease detection accuracy. The ensemble approach combines 
        the power of deep learning with traditional computer vision methods.
        """)
    
    return app

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)

