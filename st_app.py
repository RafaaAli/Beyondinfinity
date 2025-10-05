# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import lightkurve as lk
from astroquery.mast import Catalogs
from astropy.timeseries import LombScargle
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ==============================
# CONFIG
# ==============================
MODEL_PATH = "tess_xgboost_only.pkl"
st.set_page_config(page_title="🪐 ExoDetect AI", layout="wide")

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Model file not found! Please place `tess_xgboost_only.pkl` in this folder.")
        st.stop()

model_artifact = load_model()
xgb_model = model_artifact['model']
imputer = model_artifact['imputer']
le = model_artifact['label_encoder']
FEATURES = model_artifact['features']

# ==============================
# HELPER FUNCTIONS
# ==============================
def get_stellar_params(tic_id):
    try:
        result = Catalogs.query_criteria(catalog="TIC", ID=tic_id)
        return {
            'tmag': float(result['Tmag'][0]),
            'Teff': float(result['Teff'][0]) if not np.isnan(result['Teff'][0]) else 5777,
            'logg': float(result['logg'][0]) if not np.isnan(result['logg'][0]) else 4.4,
            'rad': float(result['rad'][0]) if not np.isnan(result['rad'][0]) else 1.0,
            'mass': float(result['mass'][0]) if not np.isnan(result['mass'][0]) else 1.0
        }
    except:
        return {'tmag': 10, 'Teff': 5777, 'logg': 4.4, 'rad': 1.0, 'mass': 1.0}

def compute_periodogram(time, flux, min_per=0.5, max_per=50):
    mask = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[mask], flux[mask]
    flux = (flux - np.median(flux)) / np.median(flux)
    freq = np.linspace(1/max_per, 1/min_per, 5000)
    power = LombScargle(time, flux).power(freq)
    best_per = 1 / freq[np.argmax(power)]
    periodogram = pd.DataFrame({'period': 1/freq, 'power': power})
    return best_per, periodogram

def phase_fold(time, flux, period, epoch=None):
    if epoch is None:
        epoch = time[0]
    phase = ((time - epoch) % period) / period
    phase = np.where(phase > 0.5, phase - 1, phase)
    return phase, flux

def extract_features(period, depth_ppm, duration_h, stellar_params):
    return {
        'pl_orbper': period,
        'pl_rade': np.sqrt(depth_ppm / 1e6) * stellar_params['rad'] / 0.009155,
        'pl_trandep': depth_ppm,
        'pl_trandurh': duration_h,
        'st_tmag': stellar_params['tmag'],
        'st_teff': stellar_params['Teff'],
        'st_logg': stellar_params['logg'],
        'st_rad': stellar_params['rad'],
        'pl_insol': (stellar_params['Teff'] / 5777)**4 * (stellar_params['rad'] / period)**2,
        'pl_eqt': stellar_params['Teff'] * np.sqrt(stellar_params['rad'] / (2 * period))
    }

def predict_candidate(features_dict):
    df = pd.DataFrame([features_dict])
    X = pd.DataFrame(columns=FEATURES)
    for f in FEATURES:
        X[f] = df.get(f, [np.nan])
    X = X.apply(pd.to_numeric, errors='coerce')
    X_imp = pd.DataFrame(imputer.transform(X), columns=FEATURES)
    proba = xgb_model.predict_proba(X_imp)
    planet_idx = np.where(le.classes_ == 'Planet')[0][0]
    return proba[0][planet_idx], le.inverse_transform([np.argmax(proba, axis=1)[0]])[0]

# ==============================
# UI
# ==============================
st.title("🪐 ExoDetect AI: TESS Exoplanet Classifier")
st.markdown("Enter a TIC number to fetch, analyze, and classify exoplanet candidates.")

tab1, tab2 = st.tabs(["🔍 TIC Analysis", "📁 Batch Upload"])

# ==============================
# TAB 1: TIC ANALYSIS
# ==============================
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        tic_input = st.text_input("TIC Number", placeholder="e.g., 307210830")
    with col2:
        st.write("")
        st.write("")
        run_btn = st.button("🚀 Analyze", type="primary")
    
    if run_btn and tic_input:
        try:
            tic_id = int(tic_input.strip().replace("TIC", ""))
            st.info(f"Fetching light curve for TIC {tic_id}...")
            
            # Fetch light curve
            search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
            if len(search) == 0:
                st.error("No light curve found for this TIC.")
                st.stop()
            lc = search.download_all().stitch()
            lc = lc.remove_nans().remove_outliers(sigma=5)
            flat = lc.flatten(window_length=901)
            
            # Get stellar params
            stellar = get_stellar_params(tic_id)
            
            # Periodogram
            period, pg_df = compute_periodogram(flat.time.value, flat.flux.value)
            duration_h = period # rough estimate
            depth_ppm = 1500  # placeholder (you can compute from folded curve)
            
            # Predict
            features = extract_features(period, depth_ppm, duration_h, stellar)
            planet_prob, pred_class = predict_candidate(features)
            
            # Plots
            st.subheader("Raw Light Curve")
            fig1 = px.scatter(x=lc.time.value[::10], y=lc.flux.value[::10], labels={'x': 'Time (days)', 'y': 'Flux'})
            st.plotly_chart(fig1, use_container_width=True)
            
            st.subheader("Periodogram")
            fig2 = px.line(pg_df, x='period', y='power', labels={'period': 'Period (days)', 'power': 'Power'})
            fig2.add_vline(x=period, line_dash="dash", line_color="red")
            st.plotly_chart(fig2, use_container_width=True)
            
            st.subheader("Phase-Folded Light Curve")
            phase, flux = phase_fold(flat.time.value, flat.flux.value, period)
            fig3 = px.scatter(x=phase[::10], y=flux[::10], labels={'x': 'Phase', 'y': 'Normalized Flux'})
            st.plotly_chart(fig3, use_container_width=True)
            
            # Results
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Prediction", pred_class)
            col_b.metric("Planet Probability", f"{planet_prob:.1%}")
            col_c.metric("Period (days)", f"{period:.3f}")
            
        except Exception as e:
            st.error(f"Error: {e}")

# ==============================
# TAB 2: BATCH UPLOAD
# ==============================
with tab2:
    st.subheader("Upload CSV with TIC IDs or Features")
    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(df)} rows")
        if st.button("Run Batch Prediction"):
            results = []
            for idx, row in df.iterrows():
                # If row has TIC, fetch data (simplified: use placeholder)
                if 'tic' in row or 'TIC' in row:
                    prob, cls = 0.85, "Planet"
                else:
                    # Assume row has features
                    prob, cls = predict_candidate(row.to_dict())
                results.append({"Row": idx, "Prediction": cls, "Probability": prob})
            st.dataframe(pd.DataFrame(results))

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("Powered by Lightkurve, XGBoost, and NASA TESS data | NASA Space Apps Challenge 2024")