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
from collections import deque

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

def preprocess_image(image, target_size=(128, 128), input_color="rgb"):
    """Preprocess image for prediction"""
    img = prepare_image_for_model(
        image,
        target_size=target_size[0],
        input_color=input_color,
    )
    img = np.expand_dims(img, axis=0)
    return img

def predict_gesture(model, image, labels, input_color="rgb"):
    """Make prediction and return results"""
    preprocessed = preprocess_image(
        image,
        target_size=(MODEL["img_size"], MODEL["img_size"]),
        input_color=input_color,
    )
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
    
    def __init__(self, mirror=True):
        self.model = None
        self.labels = None
        self.frame_count = 0
        self.prediction_interval = 2  # Predict every 2 frames for responsiveness
        self.last_prediction = None
        self.last_confidence = 0
        self.mirror = mirror
        self.prediction_buffer = deque(maxlen=PREDICTION["smoothing_buffer_size"])
        self.label_candidates = ["Gesture: UNCERTAIN"]
        self.fixed_frame_size = None
        self.fixed_roi_size = None
    
    def set_model(self, model, labels):
        self.model = model
        self.labels = labels
        self.label_candidates = ["Gesture: UNCERTAIN"]
        if labels:
            for label in labels:
                label_upper = label.upper()
                self.label_candidates.append(f"Gesture: {label_upper}")
                self.label_candidates.append(f"Gesture: {label_upper} (not fully certain)")
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.mirror:
            img = cv2.flip(img, 1)

        h, w = img.shape[:2]
        if self.fixed_frame_size is None:
            self.fixed_frame_size = (w, h)
            self.fixed_roi_size = min(int(PREDICTION["roi_size"] * 1.2), h, w)
        else:
            target_w, target_h = self.fixed_frame_size
            if (w, h) != self.fixed_frame_size:
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                h, w = img.shape[:2]

        roi_size = self.fixed_roi_size or min(int(PREDICTION["roi_size"] * 1.2), h, w)
        roi_x = (w - roi_size) // 2
        roi_y = (h - roi_size) // 2
        roi = img[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]
        
        # Make prediction every N frames
        if self.model is not None and self.frame_count % self.prediction_interval == 0:
            try:
                preprocessed = preprocess_image(
                    roi,
                    target_size=(MODEL["img_size"], MODEL["img_size"]),
                    input_color="bgr",
                )
                prediction = self.model.predict(preprocessed, verbose=0)[0]
                self.prediction_buffer.append(prediction)
                smoothed_prediction = np.mean(self.prediction_buffer, axis=0)

                predicted_class = np.argmax(smoothed_prediction)
                confidence = smoothed_prediction[predicted_class]

                self.last_prediction = self.labels[predicted_class]
                self.last_confidence = confidence
            except Exception as e:
                print(f"Prediction error: {e}")
        
        self.frame_count += 1
        
        # Draw ROI guide box
        roi_color = (255, 255, 255)
        if self.last_prediction:
            _, roi_color = get_prediction_display(self.last_prediction, self.last_confidence)
        cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), roi_color, 2)

        # Draw prediction on frame
        if self.last_prediction:
            font = cv2.FONT_HERSHEY_SIMPLEX
            base_font_scale_title = 1.0
            base_font_scale_conf = 0.8
            thickness_title = 2
            thickness_conf = 2
            padding_x = 12
            padding_y = 12
            line_gap = 8

            max_label_width = 0
            max_label_height = 0
            for text in self.label_candidates:
                (text_width, text_height), _ = cv2.getTextSize(
                    text,
                    font,
                    base_font_scale_title,
                    thickness_title,
                )
                max_label_width = max(max_label_width, text_width)
                max_label_height = max(max_label_height, text_height)

            conf_template = "Confidence: 100.0%"
            (conf_width, conf_height), _ = cv2.getTextSize(
                conf_template,
                font,
                base_font_scale_conf,
                thickness_conf,
            )

            base_box_width = max(max_label_width, conf_width) + padding_x * 2
            max_box_width = max(1, w - 20)
            scale = min(1.0, max_box_width / base_box_width) if base_box_width else 1.0

            font_scale_title = base_font_scale_title * scale
            font_scale_conf = base_font_scale_conf * scale
            padding_x = max(6, int(padding_x * scale))
            padding_y = max(6, int(padding_y * scale))
            line_gap = max(4, int(line_gap * scale))

            max_label_width = 0
            max_label_height = 0
            for text in self.label_candidates:
                (text_width, text_height), _ = cv2.getTextSize(
                    text,
                    font,
                    font_scale_title,
                    thickness_title,
                )
                max_label_width = max(max_label_width, text_width)
                max_label_height = max(max_label_height, text_height)

            (conf_width, conf_height), _ = cv2.getTextSize(
                conf_template,
                font,
                font_scale_conf,
                thickness_conf,
            )

            box_width = max(max_label_width, conf_width) + padding_x * 2
            box_height = padding_y * 2 + max_label_height + conf_height + line_gap
            box_x1, box_y1 = 10, 10
            box_x2 = min(box_x1 + box_width, w - 10)
            box_y2 = min(box_y1 + box_height, h - 10)

            # Draw semi-transparent background
            overlay = img.copy()
            cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (60, 60, 60), -1)
            img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
            
            # Draw text
            display_label, text_color = get_prediction_display(self.last_prediction, self.last_confidence)
            label_text = f"Gesture: {display_label}"
            confidence_text = f"Confidence: {self.last_confidence:.1%}"
            label_y = box_y1 + padding_y + max_label_height
            confidence_y = label_y + line_gap + conf_height

            cv2.putText(
                img,
                label_text,
                (box_x1 + padding_x, label_y),
                font,
                font_scale_title,
                text_color,
                thickness_title,
            )
            cv2.putText(
                img,
                confidence_text,
                (box_x1 + padding_x, confidence_y),
                font,
                font_scale_conf,
                text_color,
                thickness_conf,
            )
        
        return img

def main():
    # Header
    st.markdown('<h1 class="custom-header">🤚 Gesture AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="custom-subheader">Real-time hand gesture recognition powered by deep learning</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading AI model..."):
        model, labels = load_gesture_model()

    st.sidebar.header("Settings")
    mirror_preview = st.sidebar.toggle(
        "Mirror preview",
        value=True,
        help="Flip the camera feed horizontally to match training data.",
    )
    
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
                video_transformer_factory=lambda: VideoTransformer(mirror=mirror_preview),
                async_processing=True,
                media_stream_constraints={"video": True, "audio": False},
            )
            
            # Set model in transformer
            if ctx.video_transformer:
                ctx.video_transformer.set_model(model, labels)
            
            st.info("💡 Align your hand inside the on-screen box and keep lighting consistent for best results.")
        
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
            - Use the on-screen box as a guide
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
                        model, image_rgb, labels, input_color="rgb"
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