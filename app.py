import os
import joblib
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="IMDb Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for Premium IMDb-Inspired Cinematic Dark Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Background */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1a1e29 0%, #0b0d13 70%, #050608 100%);
        color: #f8fafc;
    }

    /* IMDb Header Styling */
    .imdb-header-container {
        background: linear-gradient(135deg, rgba(26, 30, 41, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        border: 1px solid rgba(245, 197, 24, 0.25);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .imdb-header-container::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #f5c518, #e2b616, #f5c518);
    }

    .imdb-badge {
        display: inline-block;
        background: #f5c518;
        color: #000000;
        font-weight: 900;
        font-size: 1.25rem;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        letter-spacing: -0.5px;
        vertical-align: middle;
        margin-right: 0.6rem;
        box-shadow: 0 2px 8px rgba(245, 197, 24, 0.4);
    }

    .main-title {
        display: inline-block;
        font-size: 2.3rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
        vertical-align: middle;
    }

    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.6rem;
        font-weight: 400;
    }

    /* Section Cards */
    .section-card {
        background: rgba(22, 27, 38, 0.75);
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        height: 100%;
    }

    .section-header-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f5c518;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Selected Movie Context Banner */
    .movie-banner {
        background: rgba(245, 197, 24, 0.08);
        border-left: 4px solid #f5c518;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        font-size: 0.92rem;
        color: #fef3c7;
    }

    /* Result Cards */
    .result-placeholder {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 280px;
        background: rgba(15, 23, 42, 0.4);
        border: 2px dashed rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        color: #64748b;
    }

    .result-card-positive {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.18) 0%, rgba(6, 78, 59, 0.25) 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 2.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.3);
        animation: fadeIn 0.4s ease-in-out;
        min-height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    .result-card-negative {
        background: linear-gradient(145deg, rgba(239, 68, 68, 0.18) 0%, rgba(127, 29, 29, 0.25) 100%);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 2.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.3);
        animation: fadeIn 0.4s ease-in-out;
        min-height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }

    .result-subtitle {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }

    .result-badge-positive {
        color: #34d399;
        font-size: 2.5rem;
        font-weight: 900;
        letter-spacing: -0.5px;
        margin-bottom: 0.75rem;

    }

    .result-badge-negative {
        color: #f87171;
        font-size: 2.5rem;
        font-weight: 900;
        letter-spacing: -0.5px;
        margin-bottom: 0.75rem;
    }

    .result-explanation {
        color: #f1f5f9;
        font-size: 1.1rem;
        max-width: 90%;
        line-height: 1.5;
        font-weight: 500;
    }

    /* Model Specifications Grid at Bottom */
    .spec-card {
        background: rgba(22, 27, 38, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        margin-top: 1rem;
    }

    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-top: 0.75rem;
    }

    .metric-item {
        background: rgba(11, 13, 19, 0.8);
        border: 1px solid rgba(245, 197, 24, 0.2);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }

    .metric-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f5c518;
    }

    .metric-lbl {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 0.2rem;
        font-weight: 600;
    }

    /* Streamlit Button Styling */
    div.stButton > button {
        background: linear-gradient(90deg, #f5c518 0%, #d4a017 100%) !important;
        color: #000000 !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 2rem !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 16px rgba(245, 197, 24, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
        width: 100% !important;
        margin-top: 0.5rem !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(245, 197, 24, 0.55) !important;
        background: linear-gradient(90deg, #f7d038 0%, #e2b616 100%) !important;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.97); }
        to { opacity: 1; transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)


# Cache Model Pipeline Loading
@st.cache_resource
def load_sentiment_pipeline():
    """
    Load the trained scikit-learn Pipeline (TF-IDF + LinearSVC).
    """
    possible_paths = [
        "IMDB_Sentiment_Model.pkl",
        "IMDb_Sentiment_Model.pkl",
        os.path.join(os.path.dirname(__file__), "IMDB_Sentiment_Model.pkl"),
        os.path.join(os.path.dirname(__file__), "IMDb_Sentiment_Model.pkl"),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                pipeline = joblib.load(path)
                return pipeline, path
            except Exception as e:
                st.error(f"Error loading model from {path}: {str(e)}")
                return None, None

    return None, None


def main():
    # Header Banner with IMDb branding
    st.markdown("""
    <div class="imdb-header-container">
        <div>
            <span class="imdb-badge">IMDb</span>
            <span class="main-title">Movie Review Sentiment Analyzer</span>
        </div>
        <div class="header-subtitle">
            Instant binary sentiment classification using TF-IDF feature extraction and Linear Support Vector Classifier (LinearSVC).
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load Model Pipeline
    pipeline, _ = load_sentiment_pipeline()

    if pipeline is None:
        st.error(
            "❌ Model file not found or failed to load!\n\n"
            "Please ensure `IMDB_Sentiment_Model.pkl` is located in the root directory of the application."
        )
        st.stop()

    # Predefined IMDb Movie List (Fixed UI Context)
    movie_list = [
        "The Shawshank Redemption (1994)",
        "The Godfather (1972)",
        "The Dark Knight (2008)",
        "Pulp Fiction (1994)",
        "Forrest Gump (1994)",
        "Inception (2010)",
        "Fight Club (1999)",
        "Interstellar (2014)",
        "The Matrix (1999)",
        "Goodfellas (1990)",
        "The Lord of the Rings: The Return of the King (2003)",
        "The Silence of the Lambs (1991)",
        "Saving Private Ryan (1998)",
        "Gladiator (2000)",
        "The Green Mile (1999)",
        "Parasite (2019)",
        "The Departed (2006)",
        "Whiplash (2014)",
        "The Prestige (2006)",
        "Avengers: Endgame (2019)"
    ]

    # TOP HALF: Side-by-Side Review Input (Left) & Sentiment Result (Right)
    col_input, col_result = st.columns([6, 5], gap="large")

    with col_input:
        st.markdown('<div class="section-header-title">📝 1. Enter Review</div>', unsafe_allow_html=True)

        selected_movie = st.selectbox(
            "Select Movie Context:",
            options=movie_list,
            index=0,
            help="Movie title is UI context only and is not passed as a feature to the ML model."
        )

        st.markdown(f"""
        <div class="movie-banner">
            🎬 <strong>Selected Context:</strong> {selected_movie}<br>
            <span style="color: #cbd5e1; font-size: 0.85rem;">Title provides visual context only. The model analyzes the review text exclusively.</span>
        </div>
        """, unsafe_allow_html=True)

        review_text = st.text_area(
            "Movie Review Text",
            placeholder="Write your review here... Tell us what you thought about the plot, direction, performances, and cinematography.",
            height=180,
            label_visibility="collapsed"
        )

        analyze_clicked = st.button("🔍 Analyze Sentiment", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-header-title">🎯 2. Predicted Sentiment</div>', unsafe_allow_html=True)

        # Process prediction if button was clicked
        if analyze_clicked:
            cleaned_text = review_text.strip()
            if not cleaned_text:
                st.warning("⚠️ Please write a movie review before clicking Analyze Sentiment.")
            else:
                with st.spinner("Classifying review text..."):
                    prediction = pipeline.predict([cleaned_text])[0]

                if prediction == 1:
                    st.markdown("""
                    <div class="result-card-positive">
                        <div class="result-subtitle">Classification Result</div>
                        <div class="result-badge-positive">👍 POSITIVE</div>
                        <div class="result-explanation">The review expresses an overall positive sentiment.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-card-negative">
                        <div class="result-subtitle">Classification Result</div>
                        <div class="result-badge-negative">👎 NEGATIVE</div>
                        <div class="result-explanation">The review expresses an overall negative sentiment.</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Default state before clicking analyze
            st.markdown("""
            <div class="result-placeholder">
                <div style="font-size: 3rem; margin-bottom: 0.5rem; opacity: 0.7;">🎬</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #cbd5e1; margin-bottom: 0.25rem;">Awaiting Review Input</div>
                <div style="font-size: 0.9rem;">Write a movie review on the left and click <strong>Analyze Sentiment</strong> to view the real-time prediction here.</div>
            </div>
            """, unsafe_allow_html=True)

    # BOTTOM HALF: Full-width Model Specifications & Test Metrics
    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1); margin-top: 1.5rem; margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
    st.markdown('<div class="section-header-title">📊 Model Specifications & Notebook Performance</div>', unsafe_allow_html=True)

    spec_col1, spec_col2 = st.columns([5, 6], gap="large")

    with spec_col1:
        st.markdown("""
        <div class="spec-card">
            <h4 style="margin-top:0; color:#f5c518; font-size:1.1rem;">Machine Learning Pipeline</h4>
            <ul style="padding-left: 1.2rem; color: #cbd5e1; font-size: 0.92rem; line-height: 1.7; margin-bottom:0;">
                <li><strong>Algorithm:</strong> Linear Support Vector Machine (<code>LinearSVC</code>, C=1.0)</li>
                <li><strong>Representation:</strong> <code>TF-IDF</code> (Unigrams + Bigrams, Sublinear TF)</li>
                <li><strong>Task:</strong> Binary Sentiment Classification (Positive / Negative)</li>
                <li><strong>Dataset:</strong> IMDb Large Movie Review Dataset (50,000 samples)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with spec_col2:
        st.markdown("""
        <div class="spec-card">
            <h4 style="margin-top:0; color:#f5c518; font-size:1.1rem;">Final Test Evaluation Metrics</h4>
            <p style="font-size:0.85rem; color:#94a3b8; margin-bottom: 0.5rem;">
                Evaluated on 25,000 held-out test reviews using 5-fold cross-validated hyperparameter C = 1.0.
            </p>
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-val">88.89%</div>
                    <div class="metric-lbl">Accuracy</div>
                </div>
                <div class="metric-item">
                    <div class="metric-val">89.20%</div>
                    <div class="metric-lbl">Precision</div>
                </div>
                <div class="metric-item">
                    <div class="metric-val">88.49%</div>
                    <div class="metric-lbl">Recall</div>
                </div>
                <div class="metric-item">
                    <div class="metric-val">88.84%</div>
                    <div class="metric-lbl">F1-Score</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 2rem;">
        💡 <strong>Note on Model Decision Score:</strong> <code>LinearSVC</code> optimizes a maximum-margin hyperplane hinge loss and does not output probability distributions. Predictions represent strict binary classification.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
