"""
Streamlit Web Application for News Article Classification
"""

import streamlit as st
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import BaselinePredictor, TransformerPredictor
import time


# Page configuration
st.set_page_config(
    page_title="News Article Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .world-box { background-color: #ffebee; border-left: 5px solid #f44336; }
    .sports-box { background-color: #e8f5e9; border-left: 5px solid #4caf50; }
    .business-box { background-color: #e3f2fd; border-left: 5px solid #2196f3; }
    .scitech-box { background-color: #fff3e0; border-left: 5px solid #ff9800; }
</style>
""", unsafe_allow_html=True)


# Cache model loading
@st.cache_resource
def load_baseline_model():
    """Load baseline model (cached)"""
    try:
        predictor = BaselinePredictor()
        predictor.load_model()
        return predictor, None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_transformer_model():
    """Load transformer model (cached)"""
    try:
        predictor = TransformerPredictor()
        predictor.load_model()
        return predictor, None
    except Exception as e:
        return None, str(e)


def get_category_emoji(category):
    """Get emoji for category"""
    emoji_map = {
        "World": "🌍",
        "Sports": "⚽",
        "Business": "💼",
        "Sci/Tech": "🔬"
    }
    return emoji_map.get(category, "📰")


def display_prediction(category, confidence_scores, inference_time):
    """Display prediction results"""
    # Category color mapping
    color_map = {
        "World": "world-box",
        "Sports": "sports-box",
        "Business": "business-box",
        "Sci/Tech": "scitech-box"
    }
    
    # Main prediction
    st.markdown(f"""
    <div class="prediction-box {color_map.get(category, '')}">
        <h2 style="margin: 0;">{get_category_emoji(category)} Predicted Category: <strong>{category}</strong></h2>
        <p style="margin-top: 0.5rem; font-size: 1.1rem;">Confidence: <strong>{confidence_scores[category]:.2%}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence scores for all categories
    st.subheader("📊 Confidence Scores for All Categories")
    
    # Sort by confidence
    sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
    
    for cat, score in sorted_scores:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(score)
        with col2:
            st.write(f"{get_category_emoji(cat)} **{cat}**: {score:.2%}")
    
    # Inference time
    st.info(f"⏱️ Inference time: {inference_time*1000:.2f}ms")


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">📰 News Article Topic Classifier</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <p style="text-align: center; font-size: 1.2rem; color: #666;">
        Classify news articles into <strong>World</strong>, <strong>Sports</strong>, 
        <strong>Business</strong>, or <strong>Sci/Tech</strong> categories
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_choice = st.radio(
            "Select Model:",
            ["Transformer (DistilBERT)", "Baseline (TF-IDF + LogReg)"],
            help="Choose between the transformer model (more accurate) or baseline model (faster)"
        )
        
        st.markdown("---")
        
        # About section
        st.header("ℹ️ About")
        st.markdown("""
        This application uses machine learning to automatically classify news articles.
        
        **Models:**
        - **Transformer**: DistilBERT fine-tuned on AG News
        - **Baseline**: TF-IDF + Logistic Regression
        
        **Dataset:** AG News (120K training samples)
        
        **Categories:**
        - 🌍 World
        - ⚽ Sports
        - 💼 Business
        - 🔬 Sci/Tech
        """)
        
        st.markdown("---")
        
        # Examples
        st.header("💡 Try These Examples")
        
        examples = {
            "🌍 World": "UN Security Council holds emergency meeting to discuss international crisis and humanitarian aid",
            "⚽ Sports": "Lakers defeat Boston Celtics in overtime thriller to win NBA championship game",
            "💼 Business": "Federal Reserve announces interest rate hike as inflation concerns continue to rise",
            "🔬 Sci/Tech": "New quantum computer breakthrough achieves unprecedented processing speeds and accuracy"
        }
        
        def set_example(text):
            st.session_state.text_input = text
            st.session_state.text_area_key = str(time.time())
        
        for cat, example in examples.items():
            st.button(f"{cat}", key=cat, use_container_width=True, on_click=set_example, args=(example,))
    
    # Main content area
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col2:
        # Initialize session state for text input
        if 'text_input' not in st.session_state:
            st.session_state.text_input = ''
        
        def clear_text():
            st.session_state.text_input = ''
            st.session_state.text_area_key = str(time.time())
        
        # Initialize text area key
        if 'text_area_key' not in st.session_state:
            st.session_state.text_area_key = '0'
        
        text_input = st.text_area(
            "📝 Enter a news article headline or text:",
            value=st.session_state.text_input,
            height=150,
            placeholder="Type or paste a news article here...",
            help="Enter the text you want to classify",
            key=f"text_area_{st.session_state.text_area_key}"
        )
        
        # Clear button
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
        with col_btn2:
            st.button("🗑️ Clear", use_container_width=True, on_click=clear_text)
        
        # Classify button
        if st.button("🚀 Classify Article", type="primary", use_container_width=True):
            if not text_input.strip():
                st.warning("⚠️ Please enter some text to classify.")
            else:
                with st.spinner("🔄 Classifying..."):
                    try:
                        # Load appropriate model
                        if "Transformer" in model_choice:
                            predictor, error = load_transformer_model()
                            if error:
                                st.error(f"❌ Error loading transformer model: {error}")
                                st.info("💡 Please train the transformer model first: `python src/train_transformer.py`")
                                return
                        else:
                            predictor, error = load_baseline_model()
                            if error:
                                st.error(f"❌ Error loading baseline model: {error}")
                                st.info("💡 Please train the baseline model first: `python src/train_baseline.py`")
                                return
                        
                        # Make prediction
                        start_time = time.time()
                        category, confidence_scores = predictor.predict(text_input)
                        inference_time = time.time() - start_time
                        
                        # Display results
                        st.success("✅ Classification Complete!")
                        display_prediction(category, confidence_scores, inference_time)
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        st.exception(e)
        
        st.markdown("---")
        
        # Additional information
        with st.expander("📚 How It Works"):
            st.markdown("""
            ### Classification Process
            
            1. **Text Input**: You provide a news article headline or text
            2. **Preprocessing**: The text is cleaned and prepared for the model
            3. **Model Inference**: The selected model analyzes the text
            4. **Prediction**: The model outputs the most likely category with confidence scores
            
            ### Model Details
            
            **Transformer Model (DistilBERT)**
            - Architecture: DistilBERT (66M parameters)
            - Fine-tuned on AG News dataset
            - Higher accuracy, slightly slower
            - Typical accuracy: ~94%
            
            **Baseline Model (TF-IDF + Logistic Regression)**
            - Traditional machine learning approach
            - TF-IDF vectorization (5000 features)
            - Faster inference, good baseline
            - Typical accuracy: ~87%
            
            ### Dataset
            
            Both models are trained on the **AG News dataset**:
            - 120,000 training samples
            - 7,600 test samples
            - Balanced across 4 categories
            - Source: HuggingFace Datasets
            """)
        
        with st.expander("🔧 Technical Details"):
            st.markdown("""
            ### Technology Stack
            
            - **Frontend**: Streamlit
            - **ML Framework**: PyTorch, scikit-learn
            - **Transformers**: HuggingFace Transformers
            - **Experiment Tracking**: MLflow
            - **Dataset**: HuggingFace Datasets (ag_news)
            
            ### Model Training
            
            To train the models yourself:
            
            ```bash
            # Install dependencies
            pip install -r requirements.txt
            
            # Train baseline model
            python src/train_baseline.py
            
            # Train transformer model
            python src/train_transformer.py
            
            # View experiments
            mlflow ui
            ```
            
            ### Repository
            
            Find the complete code and documentation on GitHub:
            [News-Article-classification](https://github.com/darshlukkad/News-Article-classification)
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <p style="text-align: center; color: #888; font-size: 0.9rem;">
        Built with ❤️ using Streamlit | 
        <a href="https://github.com/darshlukkad/News-Article-classification" target="_blank">GitHub</a> | 
        AG News Dataset from HuggingFace
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
