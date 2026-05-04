"""
Enhanced Streamlit App for Hand Gesture Recognition
Features:
- Real-time gesture detection from webcam
- Single image prediction mode
- Clean, minimalistic UI
- Confidence visualization
- Model info display
"""

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from PIL import Image
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

from config import (
    MODEL_PATH,
    BEST_MODEL_PATH,
    LABELS_PATH,
    GESTURES,
    MODEL,
    PREDICTION,
)
from preprocessing import prepare_image_for_model

# Page configuration
st.set_page_config(
    page_title="Gesture AI",
    page_icon="🤚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for minimalistic design
st.markdown("""
<style>
    /* Main theme */
    .main {
        padding: 2rem 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header */
    .custom-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .custom-subheader {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .prediction-label {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .prediction-confidence {
        font-size: 1.5rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Mode selector */
    .mode-tab {
        background: #f8f9fa;
        padding: 0.5rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        display: inline-block;
        font-weight: 500;
    }
    
    /* Info cards */
    .info-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Gesture pills */
    .gesture-pill {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        padding: 0.75rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_gesture_model():
    """Load model with caching"""
    model_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else MODEL_PATH
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        st.info("Please train the model first using train.py")
        st.stop()
    
    model = load_model(model_path)
    
    # Load labels
    if os.path.exists(LABELS_PATH):
        labels = np.load(LABELS_PATH).tolist()
    else:
        labels = GESTURES
    
    return model, labels

def preprocess_image(image, target_size=(128, 128)):
    """Preprocess image for prediction"""
    img = prepare_image_for_model(
        image,
        target_size=target_size[0],
        input_color="rgb",
    )
    img = np.expand_dims(img, axis=0)
    return img

def predict_gesture(model, image, labels):
    """Make prediction and return results"""
    preprocessed = preprocess_image(image, target_size=(MODEL["img_size"], MODEL["img_size"]))
    prediction = model.predict(preprocessed, verbose=0)[0]
    
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    
    return labels[predicted_class], confidence, prediction

def get_prediction_display(label, confidence):
    high_threshold = PREDICTION.get("confidence_threshold_high", PREDICTION.get("confidence_threshold", 0.7))
    low_threshold = PREDICTION.get("confidence_threshold_low", 0.5)
    color_high = PREDICTION.get("color_high_bgr", (0, 255, 0))
    color_mid = PREDICTION.get("color_mid_bgr", (0, 220, 255))
    color_low = PREDICTION.get("color_low_bgr", (0, 165, 255))

    if confidence >= high_threshold:
        return label.upper(), color_high
    if confidence >= low_threshold:
        return f"{label.upper()} (not fully certain)", color_mid
    return "UNCERTAIN", color_low

def create_confidence_chart(predictions, labels, predicted_idx):
    """Create minimalistic bar chart for prediction confidence"""
    colors = ['#667eea' if i == predicted_idx else '#e0e0e0' for i in range(len(predictions))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=predictions * 100,
            marker_color=colors,
            marker_line_color=colors,
            marker_line_width=0,
            text=[f'{p:.0f}%' for p in predictions],
            textposition='outside',
            textfont=dict(size=12, color='#333'),
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=None,
        xaxis_title=None,
        yaxis_title="Confidence (%)",
        yaxis_range=[0, 110],
        height=300,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12, color="#333"),
        margin=dict(l=20, r=20, t=20, b=40),
        xaxis=dict(
            showgrid=False,
            showline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#f0f0f0',
            showline=False,
        )
    )
    
    return fig

class VideoTransformer(VideoTransformerBase):
    """Video transformer for real-time gesture detection"""
    
    def __init__(self):
        self.model = None
        self.labels = None
        self.frame_count = 0
        self.prediction_interval = 5  # Predict every 5 frames for performance
        self.last_prediction = None
        self.last_confidence = 0
    
    def set_model(self, model, labels):
        self.model = model
        self.labels = labels
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Make prediction every N frames
        if self.model is not None and self.frame_count % self.prediction_interval == 0:
            try:
                preprocessed = preprocess_image(img_rgb, target_size=(MODEL["img_size"], MODEL["img_size"]))
                prediction = self.model.predict(preprocessed, verbose=0)[0]
                
                predicted_class = np.argmax(prediction)
                confidence = prediction[predicted_class]
                
                self.last_prediction = self.labels[predicted_class]
                self.last_confidence = confidence
            except Exception as e:
                print(f"Prediction error: {e}")
        
        self.frame_count += 1
        
        # Draw prediction on frame
        if self.last_prediction:
            # Draw semi-transparent background
            overlay = img.copy()
            cv2.rectangle(overlay, (10, 10), (400, 120), (102, 126, 234), -1)
            img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
            
            # Draw text
            display_label, text_color = get_prediction_display(self.last_prediction, self.last_confidence)
            cv2.putText(img, f"Gesture: {display_label}",
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2)
            cv2.putText(img, f"Confidence: {self.last_confidence:.1%}",
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        
        return img

def main():
    # Header
    st.markdown('<h1 class="custom-header">🤚 Gesture AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="custom-subheader">Real-time hand gesture recognition powered by deep learning</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading AI model..."):
        model, labels = load_gesture_model()
    
    # Mode selection
    st.markdown("### Choose Detection Mode")
    mode = st.radio(
        "",
        ["📹 Real-Time Detection", "📸 Single Image"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # REAL-TIME DETECTION MODE
    if mode == "📹 Real-Time Detection":
        st.markdown("### Live Camera Feed")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # WebRTC configuration
            rtc_configuration = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
            
            # Create video transformer
            ctx = webrtc_streamer(
                key="gesture-detection",
                rtc_configuration=rtc_configuration,
                video_transformer_factory=VideoTransformer,
                async_processing=True,
                media_stream_constraints={"video": True, "audio": False},
            )
            
            # Set model in transformer
            if ctx.video_transformer:
                ctx.video_transformer.set_model(model, labels)
            
            st.info("💡 Ensure your hand is clearly positioned in front of the camera. Make sure you are in a well-lit environment when using this system.")
        
        with col2:
            st.markdown("#### Supported Gestures")
            gestures_html = "".join([f'<span class="gesture-pill">{label.upper()}</span>' for label in labels])
            st.markdown(gestures_html, unsafe_allow_html=True)
            
            st.markdown("#### Tips for Best Results")
            st.markdown("""
            - ✋ Make clear, distinct gestures
            - 💡 Ensure good lighting
            - 📏 Keep hand at moderate distance
            - 🎯 Center hand in frame
            """)
    
    # SINGLE IMAGE MODE
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Upload or Capture")
            
            input_method = st.radio(
                "",
                ["📁 Upload Image", "📸 Take Photo"],
                horizontal=True,
                label_visibility="collapsed"
            )
            
            if input_method == "📸 Take Photo":
                img_file = st.camera_input("Take a picture", label_visibility="collapsed")
            else:
                img_file = st.file_uploader(
                    "Choose an image",
                    type=['jpg', 'jpeg', 'png'],
                    label_visibility="collapsed"
                )
            
            if img_file is not None:
                # Read image
                if input_method == "📸 Take Photo":
                    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = Image.open(img_file)
                    image_rgb = np.array(image)
                
                st.image(image_rgb, use_column_width=True, caption="Input Image")
        
        with col2:
            st.markdown("### Detection Results")
            
            if img_file is not None:
                # Make prediction
                with st.spinner("🔍 Analyzing gesture..."):
                    predicted_label, confidence, all_predictions = predict_gesture(
                        model, image_rgb, labels
                    )
                
                # Display prediction card
                st.markdown(f"""
                <div class="prediction-card">
                    <p class="prediction-label">{predicted_label.upper()}</p>
                    <p class="prediction-confidence">Confidence: {confidence:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence chart
                st.markdown("#### All Predictions")
                predicted_idx = np.argmax(all_predictions)
                fig = create_confidence_chart(all_predictions, labels, predicted_idx)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed breakdown
                with st.expander("📊 Detailed Breakdown"):
                    for label, prob in sorted(zip(labels, all_predictions), key=lambda x: x[1], reverse=True):
                        st.progress(float(prob), text=f"{label.upper()}: {prob:.1%}")
            else:
                st.info("👆 Upload or capture an image to begin detection")
    
    # Footer section
    st.markdown("---")
    
    with st.expander("ℹ️ About Gesture AI"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🧠 Model Information")
            st.markdown(f"""
            - **Architecture:** Convolutional Neural Network
            - **Classes:** {len(labels)} gestures
            - **Input Size:** {MODEL['img_size']}x{MODEL['img_size']} pixels
            """)
        
        with col2:
            st.markdown("#### 🎯 Supported Gestures")
            for i, label in enumerate(labels, 1):
                st.markdown(f"**{i}.** {label.upper()}")

if __name__ == "__main__":
    main()