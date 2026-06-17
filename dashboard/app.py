# dashboard/app.py
"""
Bearing Fault Detection in Induction Motors — Live Dashboard

Streams CWRU bearing data window-by-window to simulate real-time
motor health monitoring. Every number on screen is computed live
from the FFT analyzer, feature extractor, and trained classifier.
"""

import streamlit as st
import numpy as np
import pandas as pd
import scipy.io
import joblib
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Connect to your existing source code
sys.path.append(str(Path(__file__).parent.parent))
from src.fft_analyzer      import FFTAnalyzer
from src.feature_extractor import FeatureExtractor

# ─────────────────────── PAGE CONFIG ───────────────────────
st.set_page_config(
    page_title="Bearing Fault Detection — Induction Motors",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────── CONSTANTS ───────────────────────
LABEL_NAMES = ['Normal', 'Inner Race Fault', 'Outer Race Fault', 'Ball Element Fault']
COLORS      = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
SAMPLING_RATE = 12000
WINDOW_SIZE   = 4096

FILE_MAP = {
    'Normal Motor':       'data/raw/normal/97.mat',
    'Inner Race Fault':   'data/raw/inner_race/105.mat',
    'Outer Race Fault':   'data/raw/outer_race/130.mat',
    'Ball Element Fault': 'data/raw/ball/118.mat'
}

# ─────────────────────── CACHED LOADERS ───────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('models/final_model_tuned.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

@st.cache_data
def load_all_signals():
    """Load full-length signal for each fault type once."""
    signals = {}
    for name, path in FILE_MAP.items():
        if not Path(path).exists():
            continue
        mat = scipy.io.loadmat(path)
        key = [k for k in mat.keys() if not k.startswith('_')][0]
        signals[name] = mat[key].flatten()
    return signals

model, scaler = load_model()
all_signals   = load_all_signals()
analyzer      = FFTAnalyzer(SAMPLING_RATE)
extractor     = FeatureExtractor(shaft_rpm=1797)
fault_freqs   = analyzer.compute_bearing_fault_frequencies(1797)

# ─────────────────────── SIDEBAR ───────────────────────
st.sidebar.title("⚙️ Monitoring Controls")
st.sidebar.markdown("---")

selected_condition = st.sidebar.selectbox(
    "🔧 Motor Condition (Simulated)",
    list(FILE_MAP.keys())
)

window_position = st.sidebar.slider(
    "📍 Signal Position",
    min_value=0, max_value=200, value=0,
    help="Move through the recording — simulates continuous monitoring"
)

show_fault_markers = st.sidebar.checkbox(
    "Show bearing fault frequency markers", value=True
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Dataset:** CWRU Bearing Dataset\n\n"
    "**Bearing:** SKF 6205-2RS JEM\n\n"
    "**Model:** Random Forest, 200 trees\n\n"
    "**Validated Accuracy:** 92.3%\n\n"
    "All values are computed live."
)

# ─────────────────────── LIVE COMPUTATION ───────────────────────
if selected_condition not in all_signals:
    st.error(f"Data file for '{selected_condition}' not found. Check CWRU files in data/raw/")
    st.stop()

full_signal = all_signals[selected_condition]
step        = int(WINDOW_SIZE * 0.5)
max_windows = (len(full_signal) - WINDOW_SIZE) // step
start_idx   = (window_position % max(max_windows, 1)) * step
current_window = full_signal[start_idx : start_idx + WINDOW_SIZE]

# Fresh feature extraction & prediction
features       = extractor.extract_all(current_window)
feature_vector = np.array(list(features.values())).reshape(1, -1)
feature_scaled = scaler.transform(feature_vector)
prediction      = model.predict(feature_scaled)[0]
probabilities   = model.predict_proba(feature_scaled)[0]
confidence      = probabilities.max() * 100
predicted_label = LABEL_NAMES[prediction]
status_color    = COLORS[prediction]

# ─────────────────────── HEADER & KPI ROW ───────────────────────
st.title("⚙️ Bearing Fault Detection in Induction Motors")
st.caption("Live Dashboard | MCSA Methodology | Verified on CWRU Benchmark")
st.divider()

col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("🔍 Live Diagnosis", predicted_label)
with col2: st.metric("🎯 Confidence", f"{confidence:.1f}%")
with col3: st.metric("📐 Kurtosis", f"{features['kurtosis']:.2f}", delta=f"{features['kurtosis']-3.0:+.2f}")
with col4: st.metric("⬆️ Crest Factor", f"{features['crest_factor']:.2f}")
with col5: st.metric("📊 RMS", f"{features['rms']:.4f}")
st.divider()

# ─────────────────────── CHARTS (WAVEFORM & FFT) ───────────────────────
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Raw Vibration Waveform")
    t = np.arange(WINDOW_SIZE) / SAMPLING_RATE
    fig1, ax1 = plt.subplots(figsize=(7, 3.2))
    ax1.plot(t, current_window, color=status_color, linewidth=0.6)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude (g)')
    ax1.set_title(f'{selected_condition} — Window {window_position}', fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

with chart_col2:
    st.subheader("FFT Spectrum (0–600 Hz)")
    freqs, mags = analyzer.compute_fft(current_window)
    mask = freqs <= 600

    fig2, ax2 = plt.subplots(figsize=(7, 3.2))
    ax2.semilogy(freqs[mask], mags[mask], color=status_color, linewidth=0.8)

    if show_fault_markers:
        marker_colors = {'BPFO': '#c0392b', 'BPFI': '#8e44ad', 'BSF': '#2980b9'}
        for fname, fcolor in marker_colors.items():
            fval = fault_freqs[fname]
            if fval <= 600:
                ax2.axvline(fval, color=fcolor, linestyle='--', linewidth=1.2, alpha=0.75)
                ax2.text(fval+3, ax2.get_ylim()[0]*5, fname, fontsize=7, color=fcolor, rotation=90)

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Amplitude (log scale)')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

st.divider()

# ─────────────────────── FEATURE TABLE & PROBABILITIES ───────────────────────
feat_col1, feat_col2 = st.columns([1, 1])

with feat_col1:
    st.subheader("📋 Diagnostic Features (live)")
    display_features = {
        'Kurtosis':            features['kurtosis'],
        'Crest Factor':        features['crest_factor'],
        'RMS':                 features['rms'],
        'BPFI Amplitude':      features['bpfi_amp'],
        'BPFO Amplitude':      features['bpfo_amp'],
        'BSF Amplitude':       features['bsf_amp'],
        'Total Fault Energy':  features['total_fault_energy']
    }
    feat_df = pd.DataFrame(list(display_features.items()), columns=['Feature', 'Value'])
    feat_df['Value'] = feat_df['Value'].apply(lambda x: f"{x:.6f}")
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

with feat_col2:
    st.subheader("🎲 Classification Probabilities")
    prob_df = pd.DataFrame({'Class': LABEL_NAMES, 'Probability': probabilities}).sort_values('Probability', ascending=True)
    fig3, ax3 = plt.subplots(figsize=(6, 3.5))
    bar_colors = [COLORS[LABEL_NAMES.index(c)] for c in prob_df['Class']]
    ax3.barh(prob_df['Class'], prob_df['Probability'], color=bar_colors, alpha=0.85, edgecolor='black')
    ax3.set_xlim(0, 1)
    ax3.set_xlabel('Probability')
    for i, (cls, prob) in enumerate(zip(prob_df['Class'], prob_df['Probability'])):
        ax3.text(prob + 0.02, i, f'{prob:.1%}', va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

st.divider()

# ─────────────────────── HEALTH TREND ───────────────────────
st.subheader("📈 Health Trend — Last 20 Windows (live computed)")

trend_kurtosis, trend_confidence, trend_status = [], [], []

for i in range(max(0, window_position-19), window_position+1):
    idx    = (i % max(max_windows, 1)) * step
    window = full_signal[idx : idx + WINDOW_SIZE]
    feats  = extractor.extract_all(window)
    fv     = np.array(list(feats.values())).reshape(1, -1)
    fv_sc  = scaler.transform(fv)
    pred_i = model.predict(fv_sc)[0]
    
    trend_kurtosis.append(feats['kurtosis'])
    trend_confidence.append(model.predict_proba(fv_sc).max() * 100)
    trend_status.append(LABEL_NAMES[pred_i])

fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)

ax4a.plot(trend_kurtosis, marker='o', color='#8e44ad', markersize=4, linewidth=1.5)
ax4a.axhline(y=3.0, color='gray', linestyle='--', alpha=0.6, label='Healthy baseline (k=3)')
ax4a.set_ylabel('Kurtosis')
ax4a.legend(fontsize=8)
ax4a.grid(True, alpha=0.3)

ax4b.plot(trend_confidence, marker='s', color='#16a085', markersize=4, linewidth=1.5)
ax4b.set_ylabel('Confidence %')
ax4b.set_xlabel('Window (most recent → right)')
ax4b.set_ylim(0, 105)
ax4b.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig4)
plt.close(fig4)

st.caption(f"Current status sequence: {' → '.join(trend_status[-5:])}")