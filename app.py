"""
app.py — Iris Flower Classification Web Application

A professional Streamlit application that predicts Iris flower species
using a pre-trained Random Forest classifier.

Run:
    streamlit run app.py
"""

# ──────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────
import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_iris

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS — Professional dark-themed styling
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global overrides ── */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}

/* ── Main container spacing ── */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
}
.hero-banner h1 {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.hero-banner p {
    color: rgba(255,255,255,0.85);
    font-size: 1.05rem;
    margin: 0;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(145deg, #1e1e2f, #2a2a40);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}
.metric-card h3 {
    color: #a78bfa;
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.4rem;
}
.metric-card p {
    color: #ffffff;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(102, 126, 234, 0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    margin: 1rem 0;
}
.result-card .species {
    font-size: 2rem;
    font-weight: 700;
    margin: 0.5rem 0 0.3rem;
}
.result-card .confidence {
    font-size: 1.1rem;
    color: rgba(255,255,255,0.7);
}

/* ── Species-specific accent colours ── */
.setosa     { color: #34d399; }
.versicolor { color: #60a5fa; }
.virginica  { color: #f472b6; }

/* ── Info box ── */
.info-box {
    background: rgba(102, 126, 234, 0.08);
    border-left: 4px solid #667eea;
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}
.info-box p { margin: 0; color: rgba(255,255,255,0.85); }

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: rgba(255,255,255,0.4);
    font-size: 0.85rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 3rem;
}

/* ── Sidebar styling ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
section[data-testid="stSidebar"] .stSlider label {
    font-weight: 500;
}

/* ── Confidence bar ── */
.conf-bar-container {
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    overflow: hidden;
    height: 28px;
    margin: 4px 0 10px;
    position: relative;
}
.conf-bar {
    height: 100%;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 10px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #fff;
    transition: width 0.6s ease;
}
.conf-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.7);
    margin-bottom: 2px;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helper: load model
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained Iris classification model."""
    model_path = os.path.join(os.path.dirname(__file__), "model", "iris_model.pkl")
    if not os.path.exists(model_path):
        st.error(
            "⚠️ Model file not found.  \n"
            "Run `python train_model.py` first to train and save the model."
        )
        st.stop()
    return joblib.load(model_path)


@st.cache_data
def get_iris_info():
    """Return Iris dataset metadata for display purposes."""
    iris = load_iris()
    return {
        "feature_names": iris.feature_names,
        "target_names": iris.target_names,
        "data": iris.data,
        "target": iris.target,
    }


# ──────────────────────────────────────────────
# Load resources
# ──────────────────────────────────────────────
model = load_model()
iris_info = get_iris_info()

SPECIES_EMOJI = {"setosa": "🌼", "versicolor": "🌺", "virginica": "🌷"}
SPECIES_CSS   = {"setosa": "setosa", "versicolor": "versicolor", "virginica": "virginica"}
SPECIES_GRADIENT = {
    "setosa":     "linear-gradient(90deg, #34d399 0%, #059669 100%)",
    "versicolor": "linear-gradient(90deg, #60a5fa 0%, #3b82f6 100%)",
    "virginica":  "linear-gradient(90deg, #f472b6 0%, #ec4899 100%)",
}

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌸 Input Features")
    st.markdown(
        '<div class="info-box"><p>Adjust the sliders to set the flower '
        'measurements, then click <strong>Predict Species</strong>.</p></div>',
        unsafe_allow_html=True,
    )

    # Feature ranges from the dataset (min / max with small padding)
    data = iris_info["data"]
    sepal_length = st.slider(
        "Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1,
        help="Length of the sepal in centimetres",
    )
    sepal_width = st.slider(
        "Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1,
        help="Width of the sepal in centimetres",
    )
    petal_length = st.slider(
        "Petal Length (cm)", 1.0, 7.0, 4.0, 0.1,
        help="Length of the petal in centimetres",
    )
    petal_width = st.slider(
        "Petal Width (cm)", 0.1, 2.5, 1.2, 0.1,
        help="Width of the petal in centimetres",
    )

    st.markdown("---")
    predict_clicked = st.button("🔍  Predict Species", use_container_width=True, type="primary")  # noqa

    st.markdown("---")
    st.markdown("### 📖 Quick Guide")
    st.markdown(
        """
        | Feature | Typical range |
        |---------|--------------|
        | Sepal Length | 4.3 – 7.9 cm |
        | Sepal Width  | 2.0 – 4.4 cm |
        | Petal Length | 1.0 – 6.9 cm |
        | Petal Width  | 0.1 – 2.5 cm |
        """
    )

# ──────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────

# Hero banner
st.markdown(
    """
    <div class="hero-banner">
        <h1>🌸 Iris Flower Classification</h1>
        <p>Predict Iris species using a Random Forest Machine Learning model</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Project overview metrics ─────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        '<div class="metric-card"><h3>Samples</h3><p>150</p></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        '<div class="metric-card"><h3>Features</h3><p>4</p></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        '<div class="metric-card"><h3>Classes</h3><p>3</p></div>',
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        '<div class="metric-card"><h3>Algorithm</h3><p>Random Forest</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("")  # spacing

# ── Input summary ────────────────────────────
st.markdown("### 📋 Your Input")
input_df = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=["Sepal Length (cm)", "Sepal Width (cm)",
             "Petal Length (cm)", "Petal Width (cm)"],
)
st.dataframe(input_df, hide_index=True)

# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────
if predict_clicked:
    # Prepare input array
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Predict class & probabilities
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    species_name = iris_info["target_names"][prediction]
    confidence = probabilities[prediction] * 100
    emoji = SPECIES_EMOJI.get(species_name, "🌸")
    css_class = SPECIES_CSS.get(species_name, "")

    # ── Result card ──────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Prediction Result")

    st.markdown(
        f"""
        <div class="result-card">
            <p style="font-size:3rem; margin:0;">{emoji}</p>
            <p class="species {css_class}">Iris {species_name.capitalize()}</p>
            <p class="confidence">Confidence: {confidence:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Confidence breakdown ─────────────────
    st.markdown("### 📊 Confidence Breakdown")

    for idx, name in enumerate(iris_info["target_names"]):
        prob = probabilities[idx] * 100
        bar_color = SPECIES_GRADIENT.get(name, "linear-gradient(90deg, #888, #aaa)")
        sp_emoji = SPECIES_EMOJI.get(name, "🌸")

        st.markdown(
            f"""
            <div class="conf-label">
                <span>{sp_emoji} Iris {name.capitalize()}</span>
                <span>{prob:.1f}%</span>
            </div>
            <div class="conf-bar-container">
                <div class="conf-bar" style="width:{max(prob, 2):.1f}%; background:{bar_color};">
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Feature importance ───────────────────
    st.markdown("### 🔬 Feature Importance")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": [n.replace(" (cm)", "") for n in iris_info["feature_names"]],
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    st.bar_chart(feat_df.set_index("Feature"), horizontal=True, color="#667eea")

else:
    # ── Placeholder when no prediction yet ───
    st.markdown("---")
    st.markdown(
        '<div class="info-box">'
        "<p>👈 Adjust the sliders in the sidebar and click "
        "<strong>Predict Species</strong> to see the results.</p></div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# About Section
# ──────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️  About This Project", expanded=False):
    st.markdown(
        """
        **Iris Flower Classification** is a classic beginner-level machine
        learning project.  The Iris dataset—collected by biologist
        Edgar Anderson and popularised by statistician Ronald Fisher in
        1936—contains 150 samples across three species:

        | Species | Key trait |
        |---------|-----------|
        | 🌼 **Setosa** | Small petals, clearly separated |
        | 🌺 **Versicolor** | Medium-sized, overlapping features |
        | 🌷 **Virginica** | Largest petals, overlapping features |

        **Model details**

        * **Algorithm:** Random Forest Classifier (100 trees, max depth 5)
        * **Training split:** 80 / 20 stratified
        * **Evaluation:** 5-fold cross-validation

        **Tech stack:** Python · Streamlit · Scikit-learn · Pandas · NumPy · Joblib
        """
    )

with st.expander("🗂️  Dataset Preview", expanded=False):
    df = pd.DataFrame(iris_info["data"], columns=iris_info["feature_names"])
    df["Species"] = [iris_info["target_names"][t] for t in iris_info["target"]]
    st.dataframe(df, height=300)


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
        Built with ❤️ using <strong>Streamlit</strong> &amp;
        <strong>Scikit-learn</strong> &nbsp;|&nbsp;
        Iris Flower Classification &copy; 2026
    </div>
    """,
    unsafe_allow_html=True,
)
