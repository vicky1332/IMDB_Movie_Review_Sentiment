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

# Inject Custom CSS for Cinematic Dark Portfolio UI
st.markdown("""
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Background Styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }

    /* Header Styling */
    .main-header {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
        background: rgba(30, 41, 59, 0.7);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }

    .main-header h1 {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f59e0b, #ef4444, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
        max-width: 700px;
        margin: 0 auto;
    }

    /* Section Cards */
    .content-card {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 14px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        margin-bottom: 1.5rem;
    }

    /* Selected Movie Context Box */
    .movie-banner {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.15), rgba(239, 68, 68, 0.15));
        border-left: 4px solid #f59e0b;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-top: 0.5rem;
        margin-bottom: 1.25rem;
        font-size: 0.95rem;
        color: #fef3c7;
    }

    /* Result Badges */
    .result-card-positive {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.1) 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.25);
        animation: fadeIn 0.4s ease-in-out;
    }

    .result-card-negative {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.1) 100%);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.25);
        animation: fadeIn 0.4s ease-in-out;
    }

    .result-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #94a3b8;
        margin-bottom: 0.4rem;
    }

    .result-badge-positive {
        color: #34d399;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .result-badge-negative {
        color: #f87171;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .result-explanation {
        color: #e2e8f0;
        font-size: 1.05rem;
        margin-top: 0.5rem;
    }

    /* Metric Box Styling */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.75rem;
        margin-top: 1rem;
    }

    .metric-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.85rem;
        text-align: center;
    }

    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #38bdf8;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Streamlit Button Overrides */
    div.stButton > button {
        background: linear-gradient(90deg, #6366f1 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        padding: 0.65rem 2rem !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
        transition: all 0.2s ease-in-out !important;
        width: 100% !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6) !important;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


# Model Loading with Streamlit Caching
@st.cache_resource
def load_sentiment_pipeline():
    """
    Load the trained scikit-learn Pipeline (TF-IDF + LinearSVC).
    Checks relative paths for filename robustness.
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


# Main Application
def main():
    # Header Section
    st.markdown("""
    <div class="main-header">
        <h1>🎬 IMDb Movie Review Sentiment Analyzer</h1>
        <p>Analyze the sentiment of a movie review using TF-IDF feature extraction and a Linear Support Vector Classifier (LinearSVC).</p>
    </div>
    """, unsafe_allow_html=True)

    # Load Model Pipeline
    pipeline, model_path = load_sentiment_pipeline()

    if pipeline is None:
        st.error(
            "❌ Model file not found or failed to load!\n\n"
            "Please ensure `IMDB_Sentiment_Model.pkl` is located in the root directory of the application."
        )
        st.stop()

    # Predefined IMDb Movie List (Fixed UI Context List)
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

    # App Layout: 2 Columns (Main Input Column vs Model Info Sidebar/Column)
    col_main, col_info = st.columns([7, 4], gap="large")

    with col_main:
        st.subheader("1. Movie Context")
        selected_movie = st.selectbox(
            "Select a movie to review:",
            options=movie_list,
            index=0,
            help="Select a movie for context. Note: Movie title is UI context only and is not passed to the ML model."
        )

        st.markdown(f"""
        <div class="movie-banner">
            🎥 <strong>Selected Context:</strong> {selected_movie}<br>
            <small style="color: #cbd5e1;">The movie selection provides visual context. Only the review text below is analyzed by the LinearSVC model.</small>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("2. Write Movie Review")
        review_text = st.text_area(
            "Movie Review Input",
            placeholder="Write your review here... Tell us what you thought about the story, performances, direction, and cinematography.",
            height=200,
            label_visibility="collapsed"
        )

        analyze_clicked = st.button("🔍 Analyze Sentiment", use_container_width=True)

        # Prediction Processing
        if analyze_clicked:
            cleaned_text = review_text.strip()
            if not cleaned_text:
                st.warning("⚠️ Please enter a movie review before clicking Analyze Sentiment.")
            else:
                with st.spinner("Analyzing review sentiment..."):
                    # Pass ONLY the review text directly into the joblib Pipeline
                    prediction = pipeline.predict([cleaned_text])[0]

                st.markdown("### 3. Sentiment Analysis Result")

                if prediction == 1:
                    st.markdown("""
                    <div class="result-card-positive">
                        <div class="result-title">Predicted Sentiment</div>
                        <div class="result-badge-positive">👍 POSITIVE</div>
                        <div class="result-explanation">The review expresses an overall positive sentiment.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-card-negative">
                        <div class="result-title">Predicted Sentiment</div>
                        <div class="result-badge-negative">👎 NEGATIVE</div>
                        <div class="result-explanation">The review expresses an overall negative sentiment.</div>
                    </div>
                    """, unsafe_allow_html=True)

    with col_info:
        st.subheader("📊 Model Specifications")

        st.markdown("""
        <div class="content-card">
            <h4 style="margin-top:0; color:#f59e0b;">Machine Learning Architecture</h4>
            <ul style="padding-left: 1.2rem; color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">
                <li><strong>Classifier:</strong> Linear Support Vector Machine (<code>LinearSVC</code>)</li>
                <li><strong>Vectorization:</strong> <code>TF-IDF</code> (Unigrams & Bigrams)</li>
                <li><strong>Task:</strong> Binary Sentiment Classification</li>
                <li><strong>Dataset:</strong> IMDb Large Movie Review Dataset (50,000 samples)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="content-card">
            <h4 style="margin-top:0; color:#38bdf8;">Final Test Evaluation Metrics</h4>
            <p style="font-size:0.85rem; color:#94a3b8; margin-bottom: 0.5rem;">
                Evaluated on 25,000 held-out IMDb test reviews using five-fold cross-validated hyperparameter C = 1.0.
            </p>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">88.89%</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">89.20%</div>
                    <div class="metric-label">Precision</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">88.49%</div>
                    <div class="metric-label">Recall</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">88.84%</div>
                    <div class="metric-label">F1-Score</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="content-card" style="font-size: 0.85rem; color: #94a3b8;">
            💡 <strong>Technical Note:</strong> <code>LinearSVC</code> optimizes a hinge loss decision boundary and does not output calibrated probabilities. The application displays precise binary classification without fabricating confidence percentages.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
