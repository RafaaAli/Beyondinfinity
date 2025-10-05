# app.py
import streamlit as st
import numpy as np
import lightkurve as lk
from astroquery.mast import Catalogs
from leo_vetter.stellar import quadratic_ldc
from leo_vetter.main import TCELightCurve
from leo_vetter.thresholds import check_thresholds
from leo_vetter.plots import plot_summary, plot_modshift
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🪐 LEO-Vetter Web", layout="wide")
st.title("🪐 LEO-Vetter: Automated TESS Vetting")

# Input form
st.subheader("Enter Target Parameters")
col1, col2 = st.columns(2)

with col1:
    tic = st.text_input("TIC ID", value="231663901")
    per = st.number_input("Orbital Period (days)", value=1.430363, format="%.6f")
    epo = st.number_input("Transit Epoch (BJD)", value=1338.885, format="%.3f")

with col2:
    dur = st.number_input("Transit Duration (days)", value=0.069, format="%.3f")
    sector = st.number_input("Sector (optional)", value=1, min_value=1, max_value=72)

if st.button("🚀 Run LEO-Vetter", type="primary"):
    try:
        tic = int(tic)
        st.info(f"Fetching light curve for TIC {tic}...")
        
        # Download light curve
        search = lk.search_lightcurve(f"TIC {tic}", mission="TESS", author="TESS-SPOC")
        if len(search) == 0:
            st.error("No light curve found.")
            st.stop()
        lcs = search.download_all()
        lc = lcs.stitch()
        
        # Clean
        lc = lc[~np.isnan(lc["flux"]) & (lc["quality"] == 0)]
        
        # Flatten with transit mask
        transit_mask = lc.create_transit_mask(transit_time=epo, period=per, duration=dur)
        lc_flat = lc.flatten(mask=transit_mask)
        
        # Extract arrays
        time = np.asarray(lc_flat["time"].value)
        raw = np.asarray(lc["flux"].value)
        flux = np.asarray(lc_flat["flux"].value)
        flux_err = np.asarray(lc_flat["flux_err"].value)
        
        # Get stellar params
        st.info("Fetching stellar parameters...")
        
        # Fetch stellar data from TIC catalog
        result = Catalogs.query_criteria(catalog="TIC", ID=tic)
        star = {"tic": tic}

        # Required stellar parameters (with safe fallbacks)
        for key in ["rad", "mass", "Teff", "logg"]:
            val = result[key][0]
            star[key] = float(val) if not np.isnan(val) else (1.0 if key in ["mass", "rad"] else (5777 if key == "Teff" else 4.4))

        # Handle missing uncertainties (e_rad, e_Teff, etc.)
        for key in ["rad", "mass", "Teff", "logg"]:
            e_key = f"e_{key}"
            # Check if uncertainty column exists and is valid
            if e_key in result.colnames:
                e_val = result[e_key][0]
                if not np.isnan(e_val) and e_val != 0:
                    star[e_key] = float(e_val)
                else:
                    star[e_key] = 0.1 * star[key]  # 10% fallback
            else:
                star[e_key] = 0.1 * star[key]  # 10% fallback if column missing

        # ====== CRITICAL FIX: Calculate stellar density (rho) ======
        # rho = M / (4/3 * π * R^3)
        # In solar units: rho_star = M_sun / R_sun^3 ≈ 1.41 g/cm³
        # Convert to g/cm³: rho = (M/M_sun) / (R/R_sun)^3 * 1.41
        
        M_star = star["mass"]  # in solar masses
        R_star = star["rad"]   # in solar radii
        
        # Stellar density in g/cm³
        star["rho"] = (M_star / (R_star ** 3)) * 1.41  # Solar density ≈ 1.41 g/cm³
        
        # Error propagation for rho
        # δρ/ρ ≈ sqrt((δM/M)^2 + (3*δR/R)^2)
        rel_err_M = star["e_mass"] / M_star if M_star != 0 else 0.1
        rel_err_R = star["e_rad"] / R_star if R_star != 0 else 0.1
        rel_err_rho = np.sqrt(rel_err_M**2 + (3 * rel_err_R)**2)
        star["e_rho"] = star["rho"] * rel_err_rho
        # ============================================================

        # Get limb-darkening parameters
        star["u1"], star["u2"] = quadratic_ldc(star["Teff"], star["logg"])
        
        # Display stellar parameters
        with st.expander("📊 Stellar Parameters"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Radius (R☉)", f"{star['rad']:.2f} ± {star['e_rad']:.2f}")
            col2.metric("Mass (M☉)", f"{star['mass']:.2f} ± {star['e_mass']:.2f}")
            col3.metric("Density (g/cm³)", f"{star['rho']:.2f} ± {star['e_rho']:.2f}")
            
            col4, col5, col6 = st.columns(3)
            col4.metric("Teff (K)", f"{star['Teff']:.0f}")
            col5.metric("log(g)", f"{star['logg']:.2f}")
            col6.metric("Limb Dark. (u1, u2)", f"({star['u1']:.2f}, {star['u2']:.2f})")
        
        # Run LEO
        st.info("Running LEO-vetter analysis...")
        tlc = TCELightCurve(tic, time, raw, flux, flux_err, per, epo, dur, planetno=1)
        tlc.compute_flux_metrics(star, verbose=False)
        
        # Get verdict
        is_FA = check_thresholds(tlc.metrics, "FA")
        is_FP = check_thresholds(tlc.metrics, "FP")
        
        if not is_FA and not is_FP:
            verdict = "✅ Planet Candidate (PC)"
            verdict_color = "green"
        elif is_FP:
            verdict = "❌ False Positive (FP)"
            verdict_color = "red"
        else:
            verdict = "⚠️ False Alarm (FA)"
            verdict_color = "orange"
        
        st.success(verdict)
        
        # Show key metrics
        st.subheader("Key Transit Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        # Safe metric extraction with fallbacks
        try:
            depth_ppm = tlc.dep * 1e6 if hasattr(tlc, 'dep') and tlc.dep is not None else 0
            col1.metric("Transit Depth (ppm)", f"{depth_ppm:.0f}")
        except:
            col1.metric("Transit Depth (ppm)", "N/A")
        
        try:
            duration_hrs = tlc.qtran * per * 24 if hasattr(tlc, 'qtran') and tlc.qtran is not None else 0
            col2.metric("Duration (hrs)", f"{duration_hrs:.2f}")
        except:
            col2.metric("Duration (hrs)", "N/A")
        
        try:
            snr = tlc.ses if hasattr(tlc, 'ses') and tlc.ses is not None else 0
            col3.metric("SNR", f"{snr:.1f}")
        except:
            col3.metric("SNR", "N/A")
        
        try:
            rp = tlc.metrics.get('rp', None)
            if rp is not None and not np.isnan(rp):
                col4.metric("Planet Radius (R⊕)", f"{rp:.2f}")
            else:
                col4.metric("Planet Radius (R⊕)", "N/A")
        except:
            col4.metric("Planet Radius (R⊕)", "N/A")
        
        # Plot summary
        st.subheader("Vetting Summary Plot")
        fig = plt.figure(figsize=(14, 10))
        plot_summary(tlc, star, save_fig=False)
        st.pyplot(fig)
        plt.close()
        
        # Additional diagnostic plots
        st.subheader("Model Shift Diagnostic")
        fig2 = plt.figure(figsize=(10, 6))
        try:
            plot_modshift(tlc, save_fig=False)
            st.pyplot(fig2)
        except:
            st.warning("Model shift plot not available")
        plt.close()
        
        # Show metrics dict (collapsible)
        with st.expander("🔍 View All Vetting Metrics"):
            st.json(tlc.metrics)
        
        # Key flags to highlight
        st.subheader("Diagnostic Flags")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Shape & Transit Tests:**")
            shape_metrics = {
                "Odd-Even Mismatch": tlc.metrics.get("oe_sigma", "N/A"),
                "Secondary Eclipse": tlc.metrics.get("fa_signif", "N/A"),
                "V-Shape Test": tlc.metrics.get("chases_ror", "N/A"),
            }
            for key, val in shape_metrics.items():
                # Format the value safely
                try:
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        formatted_val = f"{val:.3f}"
                    else:
                        formatted_val = str(val)
                except:
                    formatted_val = "N/A"
                st.text(f"• {key}: {formatted_val}")
        
        with col2:
            st.markdown("**Centroid & Contamination:**")
            centroid_metrics = {
                "Centroid Offset": tlc.metrics.get("cent_offset", "N/A"),
                "Ghost Diagnostic": tlc.metrics.get("ghost_core", "N/A"),
                "Nearby Stars": tlc.metrics.get("num_tces", "N/A"),
            }
            for key, val in centroid_metrics.items():
                # Format the value safely
                try:
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        formatted_val = f"{val:.3f}" if isinstance(val, float) else str(val)
                    else:
                        formatted_val = str(val)
                except:
                    formatted_val = "N/A"
                st.text(f"• {key}: {formatted_val}")
                
        # Key flags to highlight
        st.subheader("Diagnostic Flags")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Flux-Based Tests:**")
            shape_metrics = {
                "Odd-Even Mismatch": tlc.metrics.get("odd_even", "N/A"),
                "Secondary Eclipse": tlc.metrics.get("sec_depth", "N/A"),
                "V-Shape Test": tlc.metrics.get("shape_corr", "N/A"),
            }
            for key, val in shape_metrics.items():
                try:
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        formatted_val = f"{val:.3f}"
                    else:
                        formatted_val = str(val)
                except:
                    formatted_val = "N/A"
                st.text(f"• {key}: {formatted_val}")

        with col2:
            st.markdown("**Pixel-Based Tests (N/A with SPOC FFI):**")
            st.text("• Centroid Offset: N/A (requires pixel data)")
            st.text("• Ghost Diagnostic: N/A (requires pixel data)")
            st.text(f"• Nearby Stars: {tlc.metrics.get('num_tces', 'N/A')}")            
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **LEO-Vetter** is an automated tool for vetting TESS transit signals.
    
    It performs multiple diagnostic tests including:
    - Odd-even transit comparison
    - Secondary eclipse search
    - Centroid motion analysis
    - V-shaped transit detection
    - Ghost diagnostic
    
    **Classification:**
    - **PC**: Planet Candidate
    - **FP**: False Positive (astrophysical)
    - **FA**: False Alarm (systematics)
    """)
    
    st.header("🔗 Quick Test Cases")
    st.markdown("""
    **Known Planet:**
    - TIC 231663901 (TOI-700 d)
    - Period: 1.430363 days
    - Epoch: 1338.885
    - Duration: 0.069 days
    
    **Eclipsing Binary:**
    - TIC 229804573
    - Try different periods
    """)
    
    st.header("📚 Resources")
    st.markdown("""
    - [LEO-Vetter GitHub](https://github.com/mkunimoto/LEO-vetter)
    - [TESS Data](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html)
    - [Lightkurve Docs](https://docs.lightkurve.org/)
    """)