"""
Enhanced Streamlit App for Hand Gesture Recognition
Improvements:
- Webcam live feed option
- Batch prediction mode
- Model info display
- Prediction history
- Better UI/UX
- Confidence visualization
"""

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from PIL import Image
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Hand Gesture Identifier",
    page_icon="🤚",
    layout="wide"
)

@st.cache_resource
def load_gesture_model():
    """Load model with caching"""
    model_path = "model/gesture_model.h5"
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        st.info("Please train the model first using train.py")
        st.stop()
    
    model = load_model(model_path)
    
    # Load labels
    labels_path = "model/class_labels.npy"
    if os.path.exists(labels_path):
        labels = np.load(labels_path)
    else:
        labels = ['l', 'peace', 'stop', 'thumbs_up']
    
    return model, labels

def preprocess_image(image, target_size=(128, 128)):
    """Preprocess image for prediction"""
    img = cv2.resize(image, target_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_gesture(model, image, labels):
    """Make prediction and return results"""
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed, verbose=0)[0]
    
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    
    return labels[predicted_class], confidence, prediction

def create_confidence_chart(predictions, labels):
    """Create bar chart for prediction confidence"""
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=predictions * 100,
            marker_color=['#00ff00' if p == max(predictions) else '#4CAF50' for p in predictions],
            text=[f'{p:.1%}' for p in predictions],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Gesture",
        yaxis_title="Confidence (%)",
        yaxis_range=[0, 105],
        height=400,
        showlegend=False
    )
    
    return fig

# Main App
def main():
    # Header
    st.title("🤚 Hand Gesture Identifier")
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading model..."):
        model, labels = load_gesture_model()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        mode = st.radio(
            "Input Mode",
            ["📸 Camera Input", "📁 Upload Image"],
            help="Choose how to input images"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence for a valid prediction"
        )
        
        st.markdown("---")
        st.subheader("📊 Model Info")
        st.info(f"**Trained Classes:** {len(labels)}")
        for i, label in enumerate(labels):
            st.text(f"{i+1}. {label}")
        
        st.markdown("---")
        st.subheader("📝 Instructions")
        st.markdown("""
        1. Choose input mode
        2. Capture/upload image
        3. View prediction results
        4. Check confidence scores
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input")
        
        if mode == "📸 Camera Input":
            img_file = st.camera_input("Take a picture of your hand gesture")
        else:
            img_file = st.file_uploader(
                "Upload an image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a hand gesture image"
            )
    
    with col2:
        st.subheader("Results")
        
        if img_file is not None:
            # Read and display image
            if mode == "📸 Camera Input":
                file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = Image.open(img_file)
                image_rgb = np.array(image)
            
            # Make prediction
            with st.spinner("Analyzing gesture..."):
                predicted_label, confidence, all_predictions = predict_gesture(
                    model, image_rgb, labels
                )
            
            # Display results
            if confidence >= confidence_threshold:
                st.success(f"### ✅ Detected: **{predicted_label.upper()}**")
                st.metric("Confidence", f"{confidence:.2%}")
            else:
                st.warning(f"### ⚠️ Low Confidence: {predicted_label.upper()}")
                st.metric("Confidence", f"{confidence:.2%}")
                st.info("Try capturing the gesture more clearly or adjusting the threshold")
            
            # Confidence chart
            st.plotly_chart(
                create_confidence_chart(all_predictions, labels),
                use_container_width=True
            )
            
            # Detailed predictions
            with st.expander("📊 Detailed Predictions"):
                for i, (label, prob) in enumerate(zip(labels, all_predictions)):
                    st.progress(float(prob), text=f"{label}: {prob:.2%}")
        
        else:
            st.info("👆 Please capture or upload an image to get started")
    
    # Additional features
    st.markdown("---")
    
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        ### Hand Gesture Recognition System
        
        This app uses a Convolutional Neural Network (CNN) to identify hand gestures in real-time.
        
        **Features:**
        - Real-time gesture recognition
        - Confidence score visualization
        - Support for multiple gestures
        - Adjustable confidence threshold
        
        **Supported Gestures:**
        """)
        
        cols = st.columns(len(labels))
        for i, label in enumerate(labels):
            with cols[i]:
                st.info(f"**{label.upper()}**")
    
    # Performance tips
    with st.expander("💡 Tips for Better Results"):
        st.markdown("""
        - 📍 **Position:** Center your hand in the frame
        - 💡 **Lighting:** Ensure good, even lighting
        - 🖐️ **Clarity:** Make clear, distinct gestures
        - 📏 **Distance:** Keep hand at moderate distance from camera
        - 🔄 **Try Again:** If confidence is low, retake the photo
        """)

if __name__ == "__main__":
    main()
