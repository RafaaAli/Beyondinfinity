# Knowledge Base: Kepler mission (2009–2018), TESS mission ongoing
# Models confirmed in your directory

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Optional imports (install if available)
try:
    import lightkurve as lk
    from astroquery.mast import Catalogs
    from leo_vetter.stellar import quadratic_ldc
    from leo_vetter.main import TCELightCurve
    from leo_vetter.thresholds import check_thresholds
    from leo_vetter.plots import plot_summary
    LEO_AVAILABLE = True
except ImportError:
    LEO_AVAILABLE = False
    st.warning("⚠️ LEO-vetter not installed. Install with: pip install lightkurve leo-vetter astroquery")

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="🪐 ExoDetect AI",
    page_icon="🪐",
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
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS — EXACT PATHS FROM YOUR DIRECTORY
# ============================================================================
@st.cache_resource
def load_models():
    models = {}
    try:
        # Kepler Models (in root)
        models['kepler_xgb'] = joblib.load('xgboost_model.pkl')
        models['kepler_lgb'] = joblib.load('lightgbm_model.pkl')
        models['kepler_mlp'] = joblib.load('mlp_model.pkl')
        models['kepler_lgb_coarse'] = joblib.load('lgb_coarse_model.pkl')
        models['kepler_xgb_coarse'] = joblib.load('xgb_coarse_model.pkl')
        models['kepler_rf'] = joblib.load('random_forest_model.pkl')
    except Exception as e:
        st.sidebar.warning(f"⚠️ Kepler models loading issue: {e}")
    
    try:
        # K2 Models (in kmodel/)
        models['k2_lgb'] = joblib.load('kmodel/lightgbm_model.pkl')
        models['k2_xgb'] = joblib.load('kmodel/xgboost_model.pkl')
    except Exception as e:
        st.sidebar.warning(f"⚠️ K2 models loading issue: {e}")
    
    return models

models = load_models()

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("🪐 ExoDetect AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔮 Quick Prediction", "🔬 TIC Analysis (LEO-Vetter)", "📊 Batch Processing", "📈 Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Available Models")
st.sidebar.info(f"""
**Kepler:** {len([k for k in models.keys() if 'kepler' in k])} models  
**K2:** {len([k for k in models.keys() if 'k2' in k])} models
""")

# ============================================================================
# FEATURE NAMES — 13 FEATURES (Kepler/K2 standard vetting features)
# ============================================================================
FEATURE_NAMES = [
    "tce_period",           # Orbital period (days)
    "tce_duration",         # Transit duration (hrs)
    "tce_depth",            # Transit depth (ppm)
    "tce_snr",              # Signal-to-noise ratio
    "tce_rp_rs",            # Planet-to-star radius ratio
    "tce_impact",           # Impact parameter
    "tce_model_chisq",      # Model chi-square
    "tce_dof",              # Degrees of freedom
    "tce_mes",              # Multiple event statistic
    "stellar_logg",         # Stellar surface gravity
    "stellar_teff",         # Stellar effective temperature (K)
    "stellar_rad",          # Stellar radius (R☉)
    "stellar_mass"          # Stellar mass (M☉)
]

# ============================================================================
# PAGE 1: HOME
# ============================================================================
if page == "🏠 Home":
    st.markdown('<p class="main-header">🪐 ExoDetect AI</p>', unsafe_allow_html=True)
    st.markdown("### *Complete Pipeline for Exoplanet Detection & Validation*")
    st.markdown("Built for **Kepler** and **K2** missions using 13-feature vetting pipeline.")
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎯 Quick Prediction")
        st.write("Enter all 13 vetting features for ML-based classification")
    with col2:
        st.markdown("### 🔬 TIC Analysis")
        st.write("Light curve + LEO-vetter validation (TESS only)")
    with col3:
        st.markdown("### 📊 Batch Processing")
        st.write("Upload CSV with 13 columns for bulk prediction")
    st.markdown("---")
    st.markdown("### 📌 Required Features (13)")
    st.code("\n".join(FEATURE_NAMES))

# ============================================================================
# PAGE 2: QUICK PREDICTION — WITH 13 INPUTS
# ============================================================================
elif page == "🔮 Quick Prediction":
    st.title("🔮 Quick Exoplanet Prediction (13 Features)")

    mission = st.selectbox("Select Mission", ["Kepler", "K2"])
    
    if mission == "Kepler":
        available_models = {
            "XGBoost": "kepler_xgb",
            "LightGBM": "kepler_lgb",
            "MLP": "kepler_mlp",
            "XGBoost (Coarse)": "kepler_xgb_coarse",
            "LightGBM (Coarse)": "kepler_lgb_coarse",
            "Random Forest": "kepler_rf"
        }
    else:  # K2
        available_models = {
            "XGBoost": "k2_xgb",
            "LightGBM": "k2_lgb"
        }
    
    model_choice = st.selectbox("Select Model", list(available_models.keys()))
    selected_model_key = available_models[model_choice]
    
    if selected_model_key not in models:
        st.error(f"❌ Model not loaded. Check sidebar for errors.")
        st.stop()
    
    st.markdown("---")
    st.markdown("### 📥 Enter All 13 Vetting Features")
    st.caption("Values must match training distribution (Kepler/K2 units)")

    # Split into two columns for better UX
    col1, col2 = st.columns(2)

    with col1:
        tce_period = st.number_input("Orbital Period (days)", value=3.52, format="%.6f")
        tce_duration = st.number_input("Transit Duration (hrs)", value=2.5, format="%.3f")
        tce_depth = st.number_input("Transit Depth (ppm)", value=1500.0, format="%.1f")
        tce_snr = st.number_input("SNR", value=12.5, format="%.2f")
        tce_rp_rs = st.number_input("Planet/Star Radius Ratio", value=0.012, format="%.5f")
        tce_impact = st.number_input("Impact Parameter", value=0.5, min_value=0.0, max_value=1.0, format="%.3f")
        tce_model_chisq = st.number_input("Model Chi-Square", value=1.2, format="%.3f")

    with col2:
        tce_dof = st.number_input("Degrees of Freedom", value=100, min_value=1)
        tce_mes = st.number_input("Multiple Event Statistic (MES)", value=10.0, format="%.2f")
        stellar_logg = st.number_input("Stellar log(g)", value=4.4, format="%.2f")
        stellar_teff = st.number_input("Stellar Teff (K)", value=5777.0, format="%.1f")
        stellar_rad = st.number_input("Stellar Radius (R☉)", value=1.0, format="%.3f")
        stellar_mass = st.number_input("Stellar Mass (M☉)", value=1.0, format="%.3f")

    if st.button("🚀 Predict", type="primary"):
        # Build feature vector in EXACT order
        features = np.array([[
            tce_period,
            tce_duration,
            tce_depth,
            tce_snr,
            tce_rp_rs,
            tce_impact,
            tce_model_chisq,
            tce_dof,
            tce_mes,
            stellar_logg,
            stellar_teff,
            stellar_rad,
            stellar_mass
        ]])
        
        model = models[selected_model_key]
        try:
            if isinstance(model, dict):
                actual_model = model['model']
                if 'imputer' in model:
                    features = model['imputer'].transform(features)
                pred = actual_model.predict(features)[0]
                proba = actual_model.predict_proba(features)[0]
                if 'label_encoder' in model:
                    pred_label = model['label_encoder'].inverse_transform([pred])[0]
                else:
                    pred_label = pred
            else:
                pred = model.predict(features)[0]
                proba = model.predict_proba(features)[0]
                pred_label = pred

            st.success("✅ Prediction Complete!")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Prediction", str(pred_label))
            with col2: st.metric("Confidence", f"{np.max(proba):.1%}")
            with col3: st.metric("Model", model_choice)

            # Class probabilities
            classes = ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"]
            prob_df = pd.DataFrame({'Class': classes[:len(proba)], 'Probability': proba})
            fig, ax = plt.subplots(figsize=(8, 3))
            colors = ['#667eea' if p == np.max(proba) else '#cccccc' for p in proba]
            ax.barh(prob_df['Class'], prob_df['Probability'], color=colors)
            ax.set_xlabel('Probability')
            ax.set_xlim([0, 1])
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.exception(e)

# ============================================================================
# PAGE 3: TIC ANALYSIS (LEO-VETTER) — TESS ONLY
# ============================================================================
# ============================================================================
# PAGE 3: TIC ANALYSIS (LEO-VETTER) — TESS ONLY
# ============================================================================
elif page == "🔬 TIC Analysis (LEO-Vetter)":
    st.title("🔬 TIC Analysis with LEO-Vetter (TESS)")
    if not LEO_AVAILABLE:
        st.error("❌ LEO-vetter not installed.")
        st.code("pip install git+https://github.com/mkunimoto/LEO-vetter.git")
        st.code("pip install git+https://github.com/stevepur/transit-diffImage.git")
        st.stop()
    
    st.info("Enter TIC ID and transit parameters to run LEO-vetter (TESS only).")
    
    col1, col2 = st.columns(2)
    with col1:
        tic = st.text_input("TIC ID", value="231663901")
        per = st.number_input("Orbital Period (days)", value=1.430363, format="%.6f")
        epo = st.number_input("Transit Epoch (BJD)", value=1338.885, format="%.3f")
    with col2:
        dur = st.number_input("Transit Duration (days)", value=0.069, format="%.3f")
        sector = st.number_input("Sector", value=1, min_value=1, max_value=72)
    
    if st.button("🚀 Run LEO-Vetter", type="primary"):
        try:
            tic_id = int(tic)
            st.info(f"Fetching TESS light curve for TIC {tic_id}...")
            
            # Download light curve
            search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="TESS-SPOC")
            if len(search) == 0:
                st.error("No light curve found.")
                st.stop()
            lcs = search.download_all()
            lc = lcs.stitch()
            lc = lc[~np.isnan(lc["flux"]) & (lc["quality"] == 0)]
            transit_mask = lc.create_transit_mask(transit_time=epo, period=per, duration=dur)
            lc_flat = lc.flatten(mask=transit_mask)
            
            # Extract arrays
            time = np.asarray(lc_flat["time"].value)
            raw = np.asarray(lc["flux"].value)
            flux = np.asarray(lc_flat["flux"].value)
            flux_err = np.asarray(lc_flat["flux_err"].value)
            
            # Get stellar params
            result = Catalogs.query_criteria(catalog="TIC", ID=tic_id)
            star = {"tic": tic_id}
            for key in ["rad", "mass", "Teff", "logg"]:
                val = result[key][0]
                star[key] = float(val) if not np.isnan(val) else (1.0 if key in ["mass", "rad"] else (5777 if key == "Teff" else 4.4))
            for key in ["rad", "mass", "Teff", "logg"]:
                e_key = f"e_{key}"
                if e_key in result.colnames:
                    e_val = result[e_key][0]
                    star[e_key] = float(e_val) if not np.isnan(e_val) and e_val != 0 else 0.1 * star[key]
                else:
                    star[e_key] = 0.1 * star[key]
            star["rho"] = (star["mass"] / (star["rad"] ** 3)) * 1.41
            rel_err_M = star["e_mass"] / star["mass"] if star["mass"] != 0 else 0.1
            rel_err_R = star["e_rad"] / star["rad"] if star["rad"] != 0 else 0.1
            rel_err_rho = np.sqrt(rel_err_M**2 + (3 * rel_err_R)**2)
            star["e_rho"] = star["rho"] * rel_err_rho
            star["u1"], star["u2"] = quadratic_ldc(star["Teff"], star["logg"])
            
            # Run LEO
            tlc = TCELightCurve(tic_id, time, raw, flux, flux_err, per, epo, dur, planetno=1)
            tlc.compute_flux_metrics(star, verbose=False)
            
            # Get verdict
            is_FA = check_thresholds(tlc.metrics, "FA")
            is_FP = check_thresholds(tlc.metrics, "FP")
            
            if not is_FA and not is_FP:
                verdict = "✅ Planet Candidate (PC)"
            elif is_FP:
                verdict = "❌ False Positive (FP)"
            else:
                verdict = "⚠️ False Alarm (FA)"
            
            st.success(verdict)
            
            # Show key metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Depth (ppm)", f"{tlc.dep * 1e6:.0f}")
            col2.metric("Duration (hrs)", f"{tlc.qtran * per * 24:.2f}")
            col3.metric("SNR", f"{tlc.ses:.1f}")
            
            # Show metrics dict
            with st.expander("🔍 View All Metrics"):
                st.json(tlc.metrics)
                
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)
# ============================================================================
# OTHER PAGES (Keep minimal for now)
# ============================================================================
elif page == "📊 Batch Processing":
    st.title("📊 Batch Processing")
    st.info("Upload a CSV with **exactly 13 columns** in this order:\n\n" + "\n".join(FEATURE_NAMES))
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if df.shape[1] != 13:
            st.error(f"Expected 13 columns, got {df.shape[1]}")
        else:
            st.write("✅ Column count correct. Add prediction logic as needed.")

elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.markdown("### Kepler Models")
    st.code("""
- XGBoost: 91% accuracy
- LightGBM: 93% accuracy
- Random Forest: 90% OOB
- MLP: 90% accuracy
    """)
    st.markdown("### K2 Models")
    st.code("""
- XGBoost: 92.5%
- LightGBM: 92.8%
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("[NASA Kepler Mission](https://www.nasa.gov/kepler)")
st.sidebar.markdown("[TESS Mission](https://tess.mit.edu/)")
st.sidebar.caption("Made with ❤️ for NASA Space Apps Challenge 2025")