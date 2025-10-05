# 🪐 ExoDetect AI

**Team BeyondInfinity**  
*NASA Space Apps Challenge 2025*

---

## 🌟 Project Overview

ExoDetect AI is a comprehensive machine learning pipeline for automated exoplanet detection and validation across NASA's Kepler, K2, and TESS missions. Our system combines state-of-the-art ML models with professional astronomical vetting tools to classify transit signals as confirmed planets, candidates, or false positives.

**Note**: This is a hackathon project with known bugs and limitations. Contributions are welcome from anyone interested in improving exoplanet detection tools!

### Challenge
[Exoplanet Detection](https://www.spaceappschallenge.org/nasa-space-apps-2025/challenges/exoplanet-detection/) - Create an AI/ML model trained on NASA's open-source exoplanet datasets with a web interface for user interaction.

The link for the app 
https://beyondinfinity-lbws.streamlit.app/
---

## 🎯 Key Features

- **Multi-Mission Support**: Models trained on Kepler, K2, and TESS datasets
- **Ensemble ML Approach**: XGBoost, LightGBM, MLP, Random Forest
- **Professional Validation**: Integration with LEO-vetter for automated signal vetting
- **Interactive Web App**: Streamlit-based interface for predictions and analysis
- **Complete Pipeline**: From TIC number input to classification with confidence scores
- **Light Curve Analysis**: Periodogram generation, phase folding, and detrending

---

## 📊 Model Performance

### Kepler Mission (9,564 samples)
- **XGBoost**: 91% accuracy, 0.981 macro AUC
- **LightGBM**: 93% accuracy, 90% precision
- **MLP Neural Network**: 90% accuracy, 0.972 macro AUC
- **Random Forest**: 90% OOB score
- **3-class problem**: CONFIRMED, CANDIDATE, FALSE POSITIVE

### K2 Mission (4,585 samples)
- **XGBoost**: 92.5% accuracy, 0.923 F1-score
- **LightGBM**: 92.8% accuracy, 0.927 F1-score
- **3-class problem**: CONFIRMED, CANDIDATE, FALSE POSITIVE

### TESS Mission (4,960 samples)
- **XGBoost**: 76% accuracy (challenging 3-class problem)
- **LightGBM**: 75% accuracy
- **TabPFN**: Experimental approach for small datasets

---

## 🚀 Our Journey


**Research & Planning**
- Explored NASA Exoplanet Archive datasets (Kepler, K2, TESS)
- Studied the transit method and common false positive types
- Discovered LEO-vetter tool for professional signal validation
- Identified class imbalance as primary challenge (1:50+ ratios)

**Data Processing & EDA**
- Downloaded cumulative Kepler catalog (9,564 KOIs)
- Downloaded K2 EPIC catalog (4,585 candidates)
- Accessed TESS TOI catalog via astroquery (4,960 objects)
- Analyzed feature distributions and missing value patterns
- Implemented robust preprocessing pipeline

**Model Development**
- Trained initial models with severe class imbalance
- Experimented with undersampling, oversampling (SMOTE)
- Optimized hyperparameters for each mission/model combination
- Discovered TabPFN effectiveness on TESS data
- Achieved breakthrough with coarse-grained models

**LEO-Vetter Integration**
- Resolved `rho` (stellar density) calculation bug
- Integrated lightkurve for TESS light curve fetching
- Connected astroquery for TIC catalog queries
- Implemented complete TIC → classification pipeline
- Generated diagnostic plots (periodograms, phase-folded curves)

**Web Application Development**
- Built Streamlit interface with 5 main pages
- Implemented 13-feature prediction system
- Added batch processing capabilities
- Created model comparison dashboard
- Debugged file path issues for deployment

**Final Polish & Documentation**
- Wrote comprehensive README
- Created demo script for judges
- Tested end-to-end workflows
- Prepared presentation materials

---

## 🛠️ Technical Architecture

### Machine Learning Pipeline

```
Raw Data (CSV)
    ↓
Preprocessing
  - Missing value imputation (median strategy)
  - Feature scaling (StandardScaler/RobustScaler)
  - Class balancing (undersampling/SMOTE)
    ↓
Model Training
  - XGBoost (gradient boosting)
  - LightGBM (fast gradient boosting)
  - MLP (neural network)
  - Random Forest (ensemble)
  - TabPFN (transformer-based, TESS)
    ↓
Validation
  - Stratified K-fold cross-validation
  - Balanced accuracy, precision, recall, F1
  - ROC-AUC for multi-class
    ↓
Deployment (joblib serialization)
```

### LEO-Vetter Integration

```
TIC Number Input
    ↓
Light Curve Fetching (lightkurve + MAST)
    ↓
Preprocessing
  - Remove NaNs and bad quality flags
  - Detrend with transit masking
    ↓
Stellar Parameter Retrieval (TIC catalog)
  - Radius, mass, temperature, surface gravity
  - Calculate stellar density (ρ = M/R³)
  - Limb darkening coefficients
    ↓
LEO-Vetter Analysis
  - Odd-even transit comparison
  - Secondary eclipse search
  - Centroid motion analysis
  - V-shaped transit detection
  - Ghost diagnostic
    ↓
Classification: PC (Planet Candidate), FP (False Positive), FA (False Alarm)
```

---

## 📁 Project Structure

```
exodetect-ai/
├── README.md
├── requirements.txt
├── st_app.py                      # Main Streamlit application
├── train_pipeline.py              # ML training script
│
├── models/
│   ├── xgboost_model.pkl          # Kepler XGBoost
│   ├── lightgbm_model.pkl         # Kepler LightGBM
│   ├── mlp_model.pkl              # Kepler MLP
│   ├── random_forest_model.pkl    # Kepler Random Forest
│   ├── lgb_coarse_model.pkl       # Kepler coarse-grained
│   ├── xgb_coarse_model.pkl       # Kepler coarse-grained
│   └── kmodel/
│       ├── xgboost_model.pkl      # K2 XGBoost
│       └── lightgbm_model.pkl     # K2 LightGBM
│
├── data/
│   ├── cumulative.csv             # Kepler dataset
│   ├── k2_epic.csv                # K2 dataset
│   └── tess_toi.csv               # TESS dataset
│
├── notebooks/
│   ├── kepler_eda.ipynb
│   ├── k2_training.ipynb
│   └── tess_tabpfn.ipynb
│
└── LEO-vetter/                    # Submodule for validation
```

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager


### Install Dependencies

```bash
# Clone repository
git clone https://github.com/BeyondInfinity/exodetect-ai.git
cd exodetect-ai

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt

# Install LEO-vetter (for TIC analysis)
pip install git+https://github.com/mkunimoto/LEO-vetter.git
pip install git+https://github.com/stevepur/transit-diffImage.git
```

### Download Datasets

```bash
# Kepler cumulative catalog
wget https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative

# K2 EPIC catalog
wget https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=k2candidates

# TESS TOI catalog (via astroquery in code)
```

---

## 🚀 Usage

### Running the Web Application

```bash
streamlit run st_app.py
```

Navigate to `http://localhost:8501`

### Quick Prediction Example

**Input (13 features):**
```python
features = {
    'tce_period': 3.52,         # days
    'tce_duration': 2.5,        # hours
    'tce_depth': 1500.0,        # ppm
    'tce_snr': 12.5,
    'tce_rp_rs': 0.012,         # radius ratio
    'tce_impact': 0.5,
    'tce_model_chisq': 1.2,
    'tce_dof': 100,
    'tce_mes': 10.0,
    'stellar_logg': 4.4,
    'stellar_teff': 5777.0,     # K
    'stellar_rad': 1.0,         # R☉
    'stellar_mass': 1.0         # M☉
}
```

**Output:**
```
Prediction: CONFIRMED
Confidence: 89.3%
Class Probabilities:
  - CANDIDATE: 8.7%
  - CONFIRMED: 89.3%
  - FALSE POSITIVE: 2.0%
```

### TIC Analysis Example

```python
# In Streamlit app
TIC: 231663901
Period: 1.430363 days
Epoch: 1338.885 BJD
Duration: 0.069 days

# Results:
LEO-Vetter: Planet Candidate (PC)
Depth: 1500 ppm
Duration: 1.66 hours
SNR: 12.5
```

### Training Custom Models

```bash
python train_pipeline.py \
    --data cumulative.csv \
    --model xgboost \
    --downsample 1000 \
    --output my_model
```

---

## 📚 Key Technologies & Libraries

### Machine Learning
- **scikit-learn** - Model training, preprocessing, metrics
- **XGBoost** - Gradient boosting (optimized for Kepler)
- **LightGBM** - Fast gradient boosting (best for K2)
- **TensorFlow/Keras** - Multi-layer perceptron
- **TabPFN** - Transformer for small TESS dataset
- **imbalanced-learn** - SMOTE, undersampling

### Astronomy & Data
- **lightkurve** - TESS/Kepler light curve analysis
- **astroquery** - MAST/TIC catalog queries
- **astropy** - FITS file handling, time series
- **LEO-vetter** - Professional signal validation

### Web Application
- **Streamlit** - Interactive web interface
- **pandas** - Data manipulation
- **matplotlib/seaborn** - Visualizations
- **joblib** - Model persistence

---

## 🎓 Scientific Background

### The Transit Method

Exoplanets are detected when they pass in front of their host star, causing a periodic dip in brightness. Key parameters:

- **Period**: Time between transits (orbital period)
- **Depth**: Fractional brightness decrease (∝ (R_p/R_*)²)
- **Duration**: Length of transit event
- **Shape**: Ingress/egress profile indicates impact parameter

### Common False Positives

1. **Eclipsing Binaries**: Two stars orbiting each other
2. **Blended Systems**: Background eclipsing binary
3. **Stellar Variability**: Spots, flares, pulsations
4. **Instrumental Artifacts**: Cosmic rays, detector noise
5. **Centroid Shifts**: Light from nearby source

### LEO-Vetter Diagnostics

- **Odd-Even Test**: Compare odd/even numbered transits
- **Secondary Eclipse**: Search for occultation signal
- **Centroid Motion**: Star position during transit
- **Ghost Diagnostic**: Nearby contaminating sources
- **Shape Analysis**: V-shaped vs U-shaped transits

---

## 📈 Results & Achievements

### Quantitative Metrics

| Mission | Samples | Models | Best Accuracy | Best F1 | ROC-AUC |
|---------|---------|--------|---------------|---------|---------|
| Kepler  | 9,564   | 3      | 93%           | 0.90    | 0.981   |
| K2      | 4,585   | 3      | 92.8%         | 0.93    | N/A     |
| TESS    | 4,960   | 3      | 76%           | 0.74    | N/A     |

### Qualitative Achievements

- Successfully handled 1:50+ class imbalance
- Integrated professional validation tools used by NASA
- Created end-to-end pipeline from raw TIC to classification
- Built intuitive interface accessible to researchers and public
- Achieved production-ready performance on Kepler data

### Challenges Overcome

1. **Class Imbalance**: Solved with strategic undersampling + coarse models
2. **Missing Values**: Robust imputation strategy (median + 10% fallback)
3. **LEO-Vetter Integration**: Fixed stellar density calculation bug
4. **TESS Difficulty**: Leveraged TabPFN for small, noisy dataset
5. **Deployment**: Resolved model path issues, created flexible architecture

---

## 🔮 Future Work

### Short-term (Next Sprint)
- Add TESS models to production app
- Implement batch CSV processing with progress bars
- Export LEO-vetter reports as PDF
- Add feature importance visualizations
- Create API endpoints for external tools

### Medium-term
- Train on TOI+ (community-vetted TESS candidates)
- Implement active learning for labeling efficiency
- Add time series visualization (interactive light curves)
- Support for custom/uploaded light curves
- Multi-mission ensemble voting

### Long-term
- Real-time TESS alert processing
- Integration with JWST follow-up planning
- Atmosphere characterization predictions
- Habitability zone calculations
- Citizen science interface for labeling

---

## 👥 Team BeyondInfinity

**Team Members:**
- **Rafaa Ali** - Co-developer and collaborator

**Roles & Contributions:**

During this intense 24-hour hackathon, we learned an immense amount about exoplanet detection, machine learning pipelines, and astronomical data processing. This project represents our first deep dive into:
- Handling severely imbalanced astronomical datasets
- Integrating professional scientific validation tools
- Building production ML pipelines from scratch
- Working with NASA's mission data archives

---

## ⚠️ Known Issues & Limitations

**Current Bugs:**
- LEO-vetter integration occasionally fails with certain TIC numbers
- Batch processing needs better error handling for malformed CSVs
- Model loading can timeout on slower connections
- Some edge cases in feature preprocessing cause prediction errors
- UI responsiveness issues with large batch uploads

**Limitations:**
- TESS models not yet integrated into production app
- No real-time validation of input feature ranges
- Limited error messages for invalid inputs
- Batch processing lacks progress tracking
- No model retraining interface

**We welcome contributions!** If you're interested in improving this tool, please see the Contributing section below.

---

## 🤝 Contributing

This is an open hackathon project and we encourage contributions from the community! Whether you're an astronomer, data scientist, or developer, there are many ways to help:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes**
4. **Test thoroughly**
5. **Commit** (`git commit -m 'Add amazing feature'`)
6. **Push** (`git push origin feature/amazing-feature`)
7. **Open a Pull Request**

### Areas That Need Help

- **Bug Fixes**: See issues labeled `bug` and `help-wanted`
- **Documentation**: Improve installation guides, add tutorials
- **Testing**: Write unit tests, integration tests
- **Features**: Implement items from Future Work section
- **UI/UX**: Improve Streamlit interface design
- **Performance**: Optimize model loading and inference
- **Data**: Add support for more missions (JWST, etc.)

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/exodetect-ai.git
cd exodetect-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run app locally
streamlit run st_app.py
```

### Code Style

- Follow PEP 8 for Python code
- Add docstrings to all functions
- Include type hints where applicable
- Write descriptive commit messages
- Add tests for new features

---

## 🎓 What We Learned

This hackathon was an incredible learning experience. Key takeaways:

1. **Class Imbalance is Hard**: Real astronomical data is heavily imbalanced (1:50+ ratios). We learned multiple strategies (undersampling, SMOTE, class weights) and when to apply each.

2. **Domain Knowledge Matters**: Understanding the physics of transits, types of false positives, and detection methods was crucial for feature engineering and model interpretation.

3. **Integration is Challenging**: Connecting our models with LEO-vetter required debugging stellar density calculations and handling missing TIC catalog data gracefully.

4. **Performance ≠ Simplicity**: Our best models often came from careful preprocessing rather than complex architectures.

5. **Time Constraints Force Prioritization**: With 24 hours, we learned to focus on MVP features and defer nice-to-haves.

6. **Open Source is Powerful**: Standing on the shoulders of giants (lightkurve, LEO-vetter, scikit-learn) let us accomplish far more than starting from scratch.

---

## 📖 References & Resources

### Datasets
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [Kepler Cumulative KOI Table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)
- [K2 EPIC Candidates](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2candidates)
- [TESS TOI Catalog](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI)

### Tools & Libraries
- [LEO-Vetter GitHub](https://github.com/mkunimoto/LEO-vetter) - Kunimoto et al. (2022)
- [Lightkurve Documentation](https://docs.lightkurve.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Scientific Papers
- Borucki et al. (2010) - *Kepler Planet-Detection Mission*
- Ricker et al. (2015) - *TESS Mission Overview*
- Kunimoto et al. (2022) - *Automated Vetting of Planet Candidates*
- Thompson et al. (2018) - *Kepler Data Characteristics Handbook*

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

Datasets are provided by NASA and are in the public domain.

---

## 🙏 Acknowledgments

- **NASA Exoplanet Science Institute** for maintaining public archives
- **Kepler/K2/TESS Science Teams** for mission data
- **Michelle Kunimoto** for LEO-vetter tool
- **Lightkurve Collaboration** for light curve analysis tools
- **NASA Space Apps Challenge** organizers
- Open-source ML community (scikit-learn, XGBoost, etc.)

---

## 📧 Contact

**Team BeyondInfinity**

- GitHub: [github.com/moe-phantom
- Email: maaabkiron@gmail.com
-LinkedIn: MOHAMED ALWTHIQ

---

## 🌌 "The cosmos is within us. We are made of star-stuff." - Carl Sagan

*Dedicated to the discovery of new worlds and the advancement of human knowledge.*

---

**Made with ❤️ for NASA Space Apps Challenge 2025**
