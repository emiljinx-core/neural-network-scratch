import streamlit as st
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# Import your NN class
from nn_model import NN, softmax, relu, sigmoid, tanh, leaky_relu

# Page config
st.set_page_config(
    page_title="MNIST Digit Classifier", 
    page_icon="✍️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #0f172a;
    }
    .stApp {
        background-color: #0f172a;
    }
    
    /* Hide streamlit header and white bar */
    header[data-testid="stHeader"] {
        display: none;
    }
    
    /* Remove top padding */
    .main .block-container {
        padding-top: 2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem 1rem;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li {
        color: white !important;
    }
    
    /* Built with section styling */
    .built-with {
        margin-top: 20px;
    }
    .built-with p {
        margin: 5px 0;
        font-size: 0.9rem;
        color: white !important;
    }
    
    /* Title styling */
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 30px rgba(59, 130, 246, 0.5);
    }
    
    /* Canvas container */
    .canvas-container {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.3);
        margin-bottom: 20px;
        display: inline-block;
    }
    
    /* Make canvas toolbar buttons visible in dark mode */
    div[class*="css"] button[title="Undo"],
    div[class*="css"] button[title="Redo"] {
        background-color: white !important;
        border: 2px solid #3b82f6 !important;
        color: #1e293b !important;
        border-radius: 5px;
        padding: 5px 10px;
    }
    
    div[class*="css"] button[title="Undo"]:hover,
    div[class*="css"] button[title="Redo"]:hover {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    
    /* Hide delete/trash button from canvas */
    button[title="Delete"] {
        display: none !important;
    }
    
    /* Prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(37, 99, 235, 0.4);
        margin: 20px 0;
    }
    
    /* Chart container */
    .chart-container {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.3);
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        with open("fitted_model_relu_optimized.pickle", "rb") as f:
            model = pickle.load(f)
        return model, True
    except FileNotFoundError:
        return None, False

model, model_loaded = load_model()

# Initialize session state for canvas clearing
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0
if 'show_prediction' not in st.session_state:
    st.session_state.show_prediction = False
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None

# SIDEBAR - Instructions and About
with st.sidebar:
    st.markdown("## 📝 Instructions")
    st.markdown("""
    1. **Draw** a digit (0-9) on canvas
    2. Click **Predict** button
    3. Click **Clear** to reset
    
    **Tip:** Draw large and centered!
    """)
    
    st.markdown("---")
    
    st.markdown("## 🧠 About")
    st.markdown("""
    Neural network built from **scratch** using only NumPy.
    
    **Architecture:**
    - Input: 784 neurons
    - Hidden: 512 → 256
    - Output: 10 neurons
    """)
    
    if model_loaded:
        st.markdown("**Performance:**")
        train_acc = model.accuracies['train'][-1]
        test_acc = model.accuracies['test'][-1]
        st.markdown(f"- Train: **{train_acc:.2f}%**")
        st.markdown(f"- Test: **{test_acc:.2f}%**")
    
    st.markdown("---")
    
    # Built with section - same style as above
    st.markdown("## ⚡ Built With")
    st.markdown("""
    **Technologies:**
    - 🐍 **NumPy** - Neural Network
    - 🎨 **Streamlit** - Web Interface
    - 📸 **OpenCV** - Image Processing
    
    _🔥 100% From Scratch!_  
    _No TensorFlow/PyTorch_
    """)

# MAIN CONTENT
st.markdown('<h1 class="main-title">✨ Handwritten Digit Recognition ✨</h1>', unsafe_allow_html=True)

if not model_loaded:
    st.error("❌ Model not found! Run `python train.py` first.")
    st.stop()

# Dynamic layout based on prediction state
if st.session_state.show_prediction and st.session_state.prediction_data is not None:
    # Two columns when prediction is shown
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        # Canvas
        st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
        st.markdown("### 🎨 Draw Your Digit")
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            background_color="white",
            update_streamlit=True,
            height=400,
            width=400,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
            display_toolbar=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)
        with btn_col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.canvas_key += 1
                st.session_state.show_prediction = False
                st.session_state.prediction_data = None
                st.rerun()
    
    with col2:
        # Show prediction
        pred_data = st.session_state.prediction_data
        
        st.markdown(f"""
        <div class="prediction-box">
            <h2 style="color: white; margin: 0; font-size: 1.8rem;">Predicted Digit</h2>
            <h1 style="color: white; font-size: 7rem; margin: 20px 0; font-weight: 900;">{int(pred_data['prediction'])}</h1>
            <p style="color: white; font-size: 1.5rem; margin: 0;">Confidence: {pred_data['confidence']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability chart
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['#2563eb' if i == int(pred_data['prediction']) else '#60a5fa' for i in range(10)]
        bars = ax.bar(range(10), pred_data['probabilities'] * 100, color=colors, edgecolor='#1e40af', linewidth=2)
        
        ax.set_xlabel("Digit", fontsize=14, fontweight='bold')
        ax.set_ylabel("Probability (%)", fontsize=14, fontweight='bold')
        ax.set_title("Probability Distribution", fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(range(10))
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add percentage labels
        for bar, prob in zip(bars, pred_data['probabilities']):
            height = bar.get_height()
            if height > 2:
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{prob*100:.0f}%',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # Centered canvas when no prediction
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Canvas
        st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
        st.markdown("### 🎨 Draw Your Digit")
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=20,
            stroke_color="black",
            background_color="white",
            update_streamlit=True,
            height=400,
            width=400,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
            display_toolbar=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)
        with btn_col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.canvas_key += 1
                st.session_state.show_prediction = False
                st.session_state.prediction_data = None
                st.rerun()

# Handle prediction
if predict_button and canvas_result.image_data is not None:
    if np.sum(canvas_result.image_data) > 0:
        # Preprocess
        img = cv2.cvtColor(canvas_result.image_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)
        img_resized = cv2.resize(img, (28, 28))
        img_inverted = 255 - img_resized
        img_normalized = img_inverted / 255.0
        img_flat = img_normalized.reshape(784, 1)
        
        # Predict
        prediction = model.predict(img_flat)
        
        # Get probabilities
        activation_functions = {
            'relu': relu,
            'sigmoid': sigmoid,
            'tanh': tanh,
            'leaky_relu': leaky_relu
        }
        activation_fn = activation_functions[model.activation]
        
        params = model.parameters
        n_layers = model.L - 1
        values = [img_flat]
        
        for l in range(1, n_layers):
            z = np.dot(params["w" + str(l)], values[l-1]) + params["b" + str(l)]
            a = activation_fn(z)
            values.append(a)
        
        z = np.dot(params["w"+str(n_layers)], values[n_layers-1]) + params["b"+str(n_layers)]
        probabilities = softmax(z).flatten()
        confidence = probabilities[int(prediction)] * 100
        
        # Store prediction data
        st.session_state.prediction_data = {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities
        }
        st.session_state.show_prediction = True
        st.rerun()
    else:
        st.warning("⚠️ Canvas is empty! Please draw a digit first.")