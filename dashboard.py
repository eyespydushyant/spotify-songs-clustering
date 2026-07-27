"""
╔══════════════════════════════════════════════════════════════════╗
║          🎵  SPOTIFY CLUSTERING DASHBOARD  🎵                   ║
║   Interactive ML-powered music analytics with K-Means           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import base64
import urllib.parse
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Clustering Dashboard",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS — Spotify-inspired premium dark UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Circular+Std:wght@300;400;700;900&family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root variables ── */
:root {
  --spotify-green:  #1DB954;
  --spotify-green2: #1ed760;
  --bg-primary:     #0A0A0F;
  --bg-card:        #111118;
  --bg-card2:       #16161f;
  --border:         rgba(29,185,84,0.15);
  --text-muted:     #b3b3b3;
  --gradient-green: linear-gradient(135deg, #1DB954 0%, #0f8a3a 100%);
  --gradient-purple: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
  --gradient-blue:   linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
  --gradient-pink:   linear-gradient(135deg, #f472b6 0%, #c026d3 100%);
  --gradient-orange: linear-gradient(135deg, #f97316 0%, #dc2626 100%);
}

/* ── Global font ── */
html, body, [class*="css"] {
  font-family: 'Inter', 'Circular Std', sans-serif !important;
}

/* ── App background ── */
.stApp { background: var(--bg-primary) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--spotify-green); border-radius: 3px; }

/* ── Hero banner ── */
.hero-banner {
  background: linear-gradient(135deg, #0f2d1a 0%, #0a1628 40%, #1a0a28 100%);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 40px 50px;
  margin-bottom: 30px;
  position: relative;
  overflow: hidden;
  animation: fadeInDown 0.7s ease;
}
.hero-banner::before {
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 260px; height: 260px;
  background: radial-gradient(circle, rgba(29,185,84,0.18) 0%, transparent 70%);
  border-radius: 50%;
}
.hero-banner::after {
  content: '';
  position: absolute;
  bottom: -40px; left: -40px;
  width: 180px; height: 180px;
  background: radial-gradient(circle, rgba(124,58,237,0.15) 0%, transparent 70%);
  border-radius: 50%;
}
.hero-title {
  font-size: 2.8rem;
  font-weight: 800;
  background: linear-gradient(90deg, #1DB954 0%, #1ed760 40%, #ffffff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0 0 8px 0;
  line-height: 1.1;
}
.hero-sub {
  font-size: 1.05rem;
  color: var(--text-muted);
  margin: 0;
  font-weight: 400;
}
.hero-badge {
  display: inline-block;
  background: rgba(29,185,84,0.15);
  border: 1px solid rgba(29,185,84,0.4);
  color: var(--spotify-green);
  padding: 5px 14px;
  border-radius: 100px;
  font-size: 0.78rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 16px;
}

/* ── Metric cards ── */
.metric-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 24px 22px;
  position: relative;
  overflow: hidden;
  transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
  animation: fadeInUp 0.6s ease both;
  cursor: default;
}
.metric-card:hover {
  transform: translateY(-4px);
  border-color: rgba(29,185,84,0.4);
  box-shadow: 0 12px 40px rgba(29,185,84,0.12);
}
.metric-card .accent-bar {
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 3px;
}
.metric-card .icon { font-size: 2rem; margin-bottom: 10px; }
.metric-card .value {
  font-size: 2rem;
  font-weight: 800;
  color: #fff;
  line-height: 1;
}
.metric-card .label {
  font-size: 0.82rem;
  color: var(--text-muted);
  margin-top: 4px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-weight: 500;
}
.metric-card .delta {
  font-size: 0.78rem;
  margin-top: 8px;
  font-weight: 600;
}
.delta-green { color: var(--spotify-green); }
.delta-blue  { color: #60a5fa; }
.delta-purple{ color: #a78bfa; }
.delta-orange{ color: #fb923c; }

/* ── Section headers ── */
.section-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 36px 0 20px 0;
}
.section-header .line {
  flex: 1;
  height: 1px;
  background: var(--border);
}
.section-header h2 {
  font-size: 1.3rem;
  font-weight: 700;
  color: #fff;
  margin: 0;
  white-space: nowrap;
}
.section-icon { font-size: 1.4rem; }

/* ── Chart cards ── */
.chart-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 24px;
  margin-bottom: 20px;
  animation: fadeInUp 0.7s ease both;
  transition: box-shadow 0.3s;
}
.chart-card:hover { box-shadow: 0 8px 32px rgba(29,185,84,0.08); }
.chart-title {
  font-size: 1rem;
  font-weight: 700;
  color: #fff;
  margin-bottom: 4px;
}
.chart-sub {
  font-size: 0.8rem;
  color: var(--text-muted);
  margin-bottom: 16px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: #0d0d14 !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stSlider > div > div > div {
  background: var(--spotify-green) !important;
}

/* ── Cluster badge pills ── */
.cluster-pill {
  display: inline-block;
  padding: 4px 14px;
  border-radius: 100px;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  margin: 3px;
}

/* ── Tag chips ── */
.tag-chip {
  display: inline-block;
  background: rgba(29,185,84,0.1);
  border: 1px solid rgba(29,185,84,0.25);
  color: var(--spotify-green);
  padding: 3px 10px;
  border-radius: 100px;
  font-size: 0.73rem;
  font-weight: 600;
  margin: 2px;
}

/* ── Animations ── */
@keyframes fadeInDown {
  from { opacity:0; transform:translateY(-20px); }
  to   { opacity:1; transform:translateY(0);     }
}
@keyframes fadeInUp {
  from { opacity:0; transform:translateY(20px); }
  to   { opacity:1; transform:translateY(0);    }
}
@keyframes pulse-green {
  0%,100% { box-shadow: 0 0 0 0 rgba(29,185,84,0.4); }
  50%      { box-shadow: 0 0 0 10px rgba(29,185,84,0); }
}
@keyframes spin-slow {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}
@keyframes blink {
  0%,100% { opacity:1; }
  50%      { opacity:0.3; }
}

/* ── Pulse dot ── */
.pulse-dot {
  display: inline-block;
  width: 8px; height: 8px;
  background: var(--spotify-green);
  border-radius: 50%;
  animation: pulse-green 2s infinite;
  margin-right: 6px;
}

/* ── Song table ── */
.song-row {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 12px 16px;
  border-radius: 12px;
  transition: background 0.2s;
  border-bottom: 1px solid rgba(255,255,255,0.04);
}
.song-row:hover { background: rgba(255,255,255,0.05); }
.song-num { color: var(--text-muted); font-size: 0.85rem; width: 24px; text-align: right; }
.song-name { flex: 1; font-weight: 500; font-size: 0.9rem; }
.song-artist { color: var(--text-muted); font-size: 0.8rem; }

/* ── Progress bar ── */
.progress-bar-bg {
  background: rgba(255,255,255,0.08);
  border-radius: 100px;
  height: 6px;
  overflow: hidden;
}
.progress-bar-fill {
  height: 100%;
  border-radius: 100px;
  transition: width 1s ease;
}

/* ── Plotly chart full width ── */
.element-container iframe { border-radius: 16px !important; }

/* ── Remove Streamlit default padding ── */
.block-container { padding: 2rem 2.5rem 2rem 2.5rem !important; }

/* ── Tooltip ── */
.tooltip-box {
  background: rgba(17,17,24,0.95);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px 16px;
  font-size: 0.82rem;
}

/* ── Waveform animation (decorative) ── */
.waveform {
  display: flex;
  align-items: center;
  gap: 3px;
  height: 28px;
}
.waveform span {
  display: block;
  width: 3px;
  background: var(--spotify-green);
  border-radius: 2px;
  animation: wave 1.2s ease-in-out infinite;
}
.waveform span:nth-child(1) { height: 40%; animation-delay: 0s; }
.waveform span:nth-child(2) { height: 80%; animation-delay: 0.1s; }
.waveform span:nth-child(3) { height: 60%; animation-delay: 0.2s; }
.waveform span:nth-child(4) { height: 100%; animation-delay: 0.3s; }
.waveform span:nth-child(5) { height: 55%; animation-delay: 0.4s; }
.waveform span:nth-child(6) { height: 85%; animation-delay: 0.5s; }
.waveform span:nth-child(7) { height: 35%; animation-delay: 0.6s; }
@keyframes wave {
  0%,100% { transform: scaleY(1);   }
  50%      { transform: scaleY(0.3);}
}

/* ── Divider ── */
.fancy-divider {
  border: none;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(29,185,84,0.4), transparent);
  margin: 28px 0;
}

/* ── Hide streamlit default elements (keep header buttons visible) ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

/* ── Sidebar labels ── */
.sidebar-label {
  font-size: 0.72rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-weight: 600;
  margin-bottom: 6px;
}

/* ── Neon glow text ── */
.neon-text {
  color: var(--spotify-green);
  text-shadow: 0 0 10px rgba(29,185,84,0.6), 0 0 30px rgba(29,185,84,0.3);
}

/* ── Spotify Card Grid ── */
.spotify-card {
  background: #181818;
  border-radius: 12px;
  padding: 16px;
  transition: background-color 0.3s ease, transform 0.3s ease, box-shadow 0.3s ease;
  position: relative;
  overflow: hidden;
  cursor: pointer;
  height: 100%;
  border: 1px solid rgba(255,255,255,0.05);
}
.spotify-card:hover {
  background-color: #282828;
  transform: translateY(-6px);
  box-shadow: 0 12px 30px rgba(0,0,0,0.5);
  border-color: rgba(29,185,84,0.3);
}
.spotify-poster-container {
  position: relative;
  width: 100%;
  padding-top: 100%; /* 1:1 Aspect Ratio */
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 12px;
  box-shadow: 0 8px 20px rgba(0,0,0,0.4);
}
.spotify-poster-img {
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 100%;
  object-fit: cover;
  transition: transform 0.4s ease;
}
.spotify-card:hover .spotify-poster-img {
  transform: scale(1.06);
}
.spotify-play-btn {
  position: absolute;
  bottom: 12px;
  right: 12px;
  width: 48px;
  height: 48px;
  background-color: var(--spotify-green);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 8px 16px rgba(0,0,0,0.4);
  opacity: 0;
  transform: translateY(12px);
  transition: all 0.3s ease;
}
.spotify-card:hover .spotify-play-btn {
  opacity: 1;
  transform: translateY(0);
}
.spotify-play-btn:hover {
  transform: scale(1.08) !important;
  background-color: var(--spotify-green2);
}
.spotify-card-title {
  font-size: 0.95rem;
  font-weight: 700;
  color: #ffffff;
  margin-bottom: 4px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.spotify-card-artist {
  font-size: 0.8rem;
  color: #b3b3b3;
  margin-bottom: 8px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.spotify-card-badge {
  display: inline-block;
  background: rgba(29,185,84,0.15);
  color: var(--spotify-green);
  font-size: 0.7rem;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 100px;
  text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
AUDIO_FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]
FEATURES_FOR_CLUSTER = [
    'danceability', 'energy', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence'
]
CLUSTER_COLORS = [
    '#1DB954', '#7c3aed', '#0ea5e9', '#f97316',
    '#f472b6', '#facc15', '#34d399', '#fb7185', '#a78bfa', '#38bdf8'
]
CLUSTER_NAMES_MAP = {
    2: ['Chill Vibes 🌙', 'High Energy 🔥'],
    3: ['Chill Vibes 🌙', 'Balanced Mix 🎼', 'High Energy 🔥'],
    4: ['Acoustic Soul 🎸', 'Chill Lounge 🌙', 'Dance Floor 💃', 'Power Anthems 🔥'],
    5: ['Acoustic Soul 🎸', 'Chill Lounge 🌙', 'Indie Dream ✨', 'Dance Floor 💃', 'Power Anthems 🔥'],
    6: ['Acoustic Soul 🎸', 'Chill Lounge 🌙', 'Indie Dream ✨', 'Vocal Jazz 🎺', 'Dance Floor 💃', 'Power Anthems 🔥'],
}


# ─────────────────────────────────────────────
#  DATA LOADING & CACHING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv('dataset.csv')
    df.dropna(subset=['track_name', 'artists'], inplace=True)
    df['duration_min'] = df['duration_ms'] / 60000
    df['loudness_norm'] = (df['loudness'] - df['loudness'].min()) / (df['loudness'].max() - df['loudness'].min())
    df['tempo_norm'] = (df['tempo'] - df['tempo'].min()) / (df['tempo'].max() - df['tempo'].min())
    return df

@st.cache_data(show_spinner=False)
def run_clustering(n_clusters: int, sample_size: int = 10000):
    df = load_data()
    sample = df.sample(n=min(sample_size, len(df)), random_state=42).copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(sample[FEATURES_FOR_CLUSTER])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    sample['cluster'] = kmeans.fit_predict(X)
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(X)
    sample['pca_x'] = coords[:, 0]
    sample['pca_y'] = coords[:, 1]
    sample['pca_z'] = coords[:, 2]
    sil = silhouette_score(X, sample['cluster'], sample_size=3000, random_state=42)
    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=FEATURES_FOR_CLUSTER
    )
    variance = pca.explained_variance_ratio_
    return sample, sil, centers, variance

@st.cache_data(show_spinner=False)
def compute_elbow(max_k: int = 10, sample_size: int = 8000):
    df = load_data()
    sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(sample[FEATURES_FOR_CLUSTER])
    inertias, sils = [], []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=8)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, labels, sample_size=2000, random_state=42))
    return list(range(2, max_k + 1)), inertias, sils


# ─────────────────────────────────────────────
#  PLOTTING HELPERS
# ─────────────────────────────────────────────
LEGEND_BASE = dict(
    bgcolor='rgba(17,17,24,0.8)',
    bordercolor='rgba(29,185,84,0.2)',
    borderwidth=1,
    font=dict(size=11),
)

def hex_to_rgba(h, alpha=0.2):
    """Convert 6-digit hex color to rgba() string (Plotly 6.x compatible)."""
    r, g, b = int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    return f'rgba({r},{g},{b},{alpha})'

# NOTE: legend is intentionally NOT in PLOTLY_LAYOUT to avoid
# "multiple values for keyword argument 'legend'" in Plotly 6.x
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color='#ffffff'),
    margin=dict(l=10, r=10, t=40, b=10),
)


def grid_style():
    return dict(
        gridcolor='rgba(255,255,255,0.06)',
        zerolinecolor='rgba(255,255,255,0.08)',
        tickfont=dict(size=10, color='#888'),
    )


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 10px;'>
      <div style='font-size:2.5rem;'>🎵</div>
      <div style='font-size:1.1rem; font-weight:800; color:#1DB954; letter-spacing:0.02em;'>SpotiCluster</div>
      <div style='font-size:0.72rem; color:#b3b3b3; margin-top:3px;'>ML Music Analytics</div>
    </div>
    <hr style='border:none; height:1px; background:rgba(29,185,84,0.2); margin:16px 0;'>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sidebar-label">🔬 Clustering</p>', unsafe_allow_html=True)
    n_clusters = st.slider("Number of Clusters (K)", 2, 8, 4, 1)
    sample_size = st.select_slider(
        "Sample Size",
        options=[2000, 5000, 10000, 20000, 30000],
        value=10000,
        help="Larger = slower but more representative"
    )

    st.markdown('<hr style="border:none;height:1px;background:rgba(29,185,84,0.1);margin:18px 0;">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">🎛️ Filters</p>', unsafe_allow_html=True)

    df_full = load_data()
    all_genres = sorted(df_full['track_genre'].unique().tolist())
    selected_genres = st.multiselect(
        "Filter by Genre",
        options=all_genres,
        default=[],
        placeholder="All genres"
    )

    pop_range = st.slider("Popularity Range", 0, 100, (0, 100))
    show_explicit = st.checkbox("Include Explicit Tracks", value=True)

    st.markdown('<hr style="border:none;height:1px;background:rgba(29,185,84,0.1);margin:18px 0;">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">🎨 Visualization</p>', unsafe_allow_html=True)
    chart_height = st.slider("Chart Height (px)", 350, 700, 480, 50)
    show_3d = st.checkbox("Enable 3D Cluster Plot", value=True)
    show_elbow = st.checkbox("Show Elbow Analysis", value=True)

    st.markdown("""
    <hr style='border:none;height:1px;background:rgba(29,185,84,0.1);margin:18px 0;'>
    <div style='font-size:0.72rem;color:#555;text-align:center;padding-bottom:10px;'>
      Built with ❤️ using Streamlit + Plotly<br>
      K-Means · PCA · Silhouette Analysis
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD DATA & RUN CLUSTERING
# ─────────────────────────────────────────────
with st.spinner("🎵 Loading & clustering your Spotify universe..."):
    df_raw = load_data()

    # Apply sidebar filters
    df_filtered = df_raw.copy()
    if selected_genres:
        df_filtered = df_filtered[df_filtered['track_genre'].isin(selected_genres)]
    df_filtered = df_filtered[
        (df_filtered['popularity'] >= pop_range[0]) &
        (df_filtered['popularity'] <= pop_range[1])
    ]
    if not show_explicit:
        df_filtered = df_filtered[~df_filtered['explicit']]

    # Run clustering on full (unfiltered) dataset for consistency
    clustered, sil_score, cluster_centers, pca_variance = run_clustering(n_clusters, sample_size)

    # Merge cluster labels back where possible
    cluster_names = CLUSTER_NAMES_MAP.get(n_clusters, [f"Cluster {i}" for i in range(n_clusters)])


# ─────────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="hero-banner">
  <div class="hero-badge">
    <span class="pulse-dot"></span>Live Analysis • {len(df_raw):,} Tracks
  </div>
  <h1 class="hero-title">Spotify Clustering<br>Dashboard</h1>
  <p class="hero-sub">
    K-Means ML clustering on <strong>{sample_size:,}</strong> tracks across <strong>{df_raw['track_genre'].nunique()}</strong> genres
    &nbsp;·&nbsp; Silhouette Score: <span style="color:#1DB954; font-weight:700;">{sil_score:.3f}</span>
    &nbsp;·&nbsp; {n_clusters} Clusters Identified
  </p>
  <div style="display:flex; align-items:center; gap:10px; margin-top:20px;">
    <div class="waveform">
      <span></span><span></span><span></span><span></span>
      <span></span><span></span><span></span>
    </div>
    <span style="font-size:0.8rem; color:#b3b3b3;">Analyzing audio DNA...</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  KPI METRICS ROW
# ─────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

metrics = [
    {
        "icon": "🎵", "value": f"{len(df_raw):,}", "label": "Total Tracks",
        "delta": f"114 Genres", "delta_class": "delta-green",
        "gradient": "var(--gradient-green)"
    },
    {
        "icon": "🎭", "value": str(n_clusters), "label": "Clusters Found",
        "delta": f"K-Means Algorithm", "delta_class": "delta-purple",
        "gradient": "var(--gradient-purple)"
    },
    {
        "icon": "📊", "value": f"{sil_score:.3f}", "label": "Silhouette Score",
        "delta": "Cluster quality" + (" ✓ Good" if sil_score > 0.2 else " ⚠ Low"),
        "delta_class": "delta-green" if sil_score > 0.2 else "delta-orange",
        "gradient": "var(--gradient-blue)"
    },
    {
        "icon": "⚡", "value": f"{df_raw['energy'].mean():.2f}", "label": "Avg Energy",
        "delta": f"Max: {df_raw['energy'].max():.2f}", "delta_class": "delta-orange",
        "gradient": "var(--gradient-orange)"
    },
    {
        "icon": "💃", "value": f"{df_raw['danceability'].mean():.2f}", "label": "Avg Danceability",
        "delta": f"Valence: {df_raw['valence'].mean():.2f}", "delta_class": "delta-blue",
        "gradient": "var(--gradient-pink)"
    },
]

for col, m in zip([col1, col2, col3, col4, col5], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="accent-bar" style="background:{m['gradient']};"></div>
          <div class="icon">{m['icon']}</div>
          <div class="value">{m['value']}</div>
          <div class="label">{m['label']}</div>
          <div class="delta {m['delta_class']}">{m['delta']}</div>
        </div>
        """, unsafe_allow_html=True)


st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SECTION 1: CLUSTER SCATTER (PCA)
# ─────────────────────────────────────────────
def section_header(icon, title):
    st.markdown(f"""
    <div class="section-header">
      <span class="section-icon">{icon}</span>
      <h2>{title}</h2>
      <div class="line"></div>
    </div>
    """, unsafe_allow_html=True)

section_header("🔵", "Cluster Visualization")

clustered['cluster_name'] = clustered['cluster'].apply(
    lambda x: cluster_names[x] if x < len(cluster_names) else f"Cluster {x}"
)

c1, c2 = st.columns([3, 2])

with c1:
    st.markdown('<div class="chart-card"><div class="chart-title">PCA 2D Cluster Map</div><div class="chart-sub">Principal Component Analysis — audio feature space projection</div>', unsafe_allow_html=True)
    fig_scatter = go.Figure()
    for i in range(n_clusters):
        mask = clustered['cluster'] == i
        sub = clustered[mask]
        name = cluster_names[i] if i < len(cluster_names) else f"Cluster {i}"
        fig_scatter.add_trace(go.Scatter(
            x=sub['pca_x'], y=sub['pca_y'],
            mode='markers',
            name=name,
            marker=dict(
                color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                size=5,
                opacity=0.7,
                line=dict(width=0),
            ),
            text=sub['track_name'].str[:30] + ' — ' + sub['artists'].str[:20],
            hovertemplate='<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>',
        ))
    fig_scatter.update_layout(
        **PLOTLY_LAYOUT,
        height=chart_height,
        xaxis=dict(title='PC 1', **grid_style()),
        yaxis=dict(title='PC 2', **grid_style()),
        title=dict(text='', x=0),
    )
    st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="chart-card"><div class="chart-title">Cluster Distribution</div><div class="chart-sub">Track count per cluster segment</div>', unsafe_allow_html=True)
    cluster_counts = clustered['cluster_name'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    fig_donut = go.Figure(go.Pie(
        labels=cluster_counts['Cluster'],
        values=cluster_counts['Count'],
        hole=0.62,
        marker=dict(
            colors=CLUSTER_COLORS[:n_clusters],
            line=dict(color='#0A0A0F', width=3),
        ),
        textfont=dict(size=11, color='white'),
        hovertemplate='<b>%{label}</b><br>Tracks: %{value:,}<br>Share: %{percent}<extra></extra>',
    ))
    fig_donut.add_annotation(
        text=f"<b>{len(clustered):,}</b><br><span style='font-size:10px'>tracks</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color='white'),
        align='center',
    )
    fig_donut.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#ffffff'),
        margin=dict(l=10, r=10, t=40, b=10),
        height=chart_height,
        showlegend=True,
        legend=dict(**LEGEND_BASE, orientation='v', x=1.01, y=0.5),
    )
    st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SECTION 2: 3D CLUSTER PLOT
# ─────────────────────────────────────────────
if show_3d:
    section_header("🌐", "3D Audio Feature Space")
    st.markdown('<div class="chart-card"><div class="chart-title">3D PCA Cluster Visualization</div><div class="chart-sub">Rotate & zoom to explore the three principal dimensions of audio DNA</div>', unsafe_allow_html=True)

    fig_3d = go.Figure()
    sample_3d = clustered.sample(n=min(3000, len(clustered)), random_state=42)
    for i in range(n_clusters):
        mask = sample_3d['cluster'] == i
        sub = sample_3d[mask]
        name = cluster_names[i] if i < len(cluster_names) else f"Cluster {i}"
        fig_3d.add_trace(go.Scatter3d(
            x=sub['pca_x'], y=sub['pca_y'], z=sub['pca_z'],
            mode='markers',
            name=name,
            marker=dict(
                color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                size=3, opacity=0.75,
                line=dict(width=0),
            ),
            text=sub['track_name'].str[:25],
            hovertemplate='<b>%{text}</b><extra></extra>',
        ))
    fig_3d.update_layout(
        **PLOTLY_LAYOUT,
        height=580,
        scene=dict(
            xaxis=dict(title='PC 1', gridcolor='rgba(255,255,255,0.06)', backgroundcolor='rgba(0,0,0,0)', showbackground=True),
            yaxis=dict(title='PC 2', gridcolor='rgba(255,255,255,0.06)', backgroundcolor='rgba(0,0,0,0)', showbackground=True),
            zaxis=dict(title='PC 3', gridcolor='rgba(255,255,255,0.06)', backgroundcolor='rgba(0,0,0,0)', showbackground=True),
            bgcolor='rgba(10,10,15,1)',
        ),
    )
    st.plotly_chart(fig_3d, use_container_width=True, config={'displayModeBar': True})
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SECTION 3: RADAR CHARTS (Audio DNA per Cluster)
# ─────────────────────────────────────────────
section_header("🕸️", "Cluster Audio DNA — Radar Charts")

radar_features = ['danceability', 'energy', 'speechiness', 'acousticness',
                  'instrumentalness', 'liveness', 'valence']
radar_labels = ['Dance', 'Energy', 'Speech', 'Acoustic', 'Instrumental', 'Live', 'Valence']

st.markdown('<div class="chart-card"><div class="chart-title">Audio Feature Radar by Cluster</div><div class="chart-sub">Normalized 0–1 scale — each axis represents an audio attribute</div>', unsafe_allow_html=True)
fig_radar = go.Figure()
for i in range(n_clusters):
    vals = cluster_centers.iloc[i][radar_features].tolist()
    # Normalize to 0-1
    vals_norm = [(v - df_raw[f].min()) / (df_raw[f].max() - df_raw[f].min() + 1e-9) for v, f in zip(vals, radar_features)]
    vals_norm += [vals_norm[0]]
    lbs = radar_labels + [radar_labels[0]]
    name = cluster_names[i] if i < len(cluster_names) else f"Cluster {i}"
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_norm, theta=lbs,
        fill='toself',
        name=name,
        line=dict(color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], width=2),
        fillcolor=hex_to_rgba(CLUSTER_COLORS[i % len(CLUSTER_COLORS)], 0.12),
        opacity=0.9,
    ))

fig_radar.update_layout(
    **PLOTLY_LAYOUT,
    polar=dict(
        bgcolor='rgba(0,0,0,0)',
        angularaxis=dict(
            tickfont=dict(size=11, color='#ddd'),
            gridcolor='rgba(255,255,255,0.08)',
            linecolor='rgba(255,255,255,0.1)',
        ),
        radialaxis=dict(
            visible=True,
            range=[0, 1],
            tickfont=dict(size=8, color='#666'),
            gridcolor='rgba(255,255,255,0.08)',
            linecolor='rgba(255,255,255,0.08)',
        ),
    ),
    height=500,
    showlegend=True,
)
st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})
st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SECTION 4: CORRELATION HEATMAP + FEATURE DIST
# ─────────────────────────────────────────────
section_header("🔥", "Feature Analysis")

col_heat, col_box = st.columns([1, 1])

with col_heat:
    st.markdown('<div class="chart-card"><div class="chart-title">Audio Feature Correlation Matrix</div><div class="chart-sub">Pearson correlation between all numeric features</div>', unsafe_allow_html=True)
    corr_features = ['danceability','energy','loudness_norm','speechiness',
                     'acousticness','instrumentalness','liveness','valence','tempo_norm','popularity']
    corr_labels   = ['Dance','Energy','Loudness','Speech','Acoustic','Instrum','Live','Valence','Tempo','Popularity']
    sample_corr = df_raw.sample(n=min(20000, len(df_raw)), random_state=42)
    corr_matrix = sample_corr[corr_features].corr().values

    fig_heat = go.Figure(go.Heatmap(
        z=corr_matrix,
        x=corr_labels, y=corr_labels,
        colorscale=[
            [0.0, '#1a0533'], [0.25, '#4f1b8c'], [0.5, '#0A0A0F'],
            [0.75, '#0f5c2c'], [1.0, '#1DB954']
        ],
        zmid=0,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont=dict(size=9, color='white'),
        hovertemplate='<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.3f}<extra></extra>',
        showscale=True,
        colorbar=dict(
            tickfont=dict(color='#888', size=9),
            outlinewidth=0,
            thickness=12,
        ),
    ))
    fig_heat.update_layout(
        **PLOTLY_LAYOUT,
        height=chart_height,
        xaxis=dict(tickfont=dict(size=9, color='#aaa'), side='bottom'),
        yaxis=dict(tickfont=dict(size=9, color='#aaa'), autorange='reversed'),
    )
    st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_box:
    st.markdown('<div class="chart-card"><div class="chart-title">Energy vs Danceability by Genre</div><div class="chart-sub">Scatter density — top 12 genres by popularity</div>', unsafe_allow_html=True)
    top_genres = df_raw.groupby('track_genre')['popularity'].mean().nlargest(12).index.tolist()
    df_top = df_raw[df_raw['track_genre'].isin(top_genres)].sample(n=min(5000, len(df_raw)), random_state=42)
    fig_scatter2 = px.scatter(
        df_top, x='energy', y='danceability',
        color='track_genre',
        size='popularity',
        size_max=14,
        opacity=0.7,
        hover_data={'track_name': True, 'artists': True, 'popularity': True, 'track_genre': True},
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )
    fig_scatter2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#ffffff'),
        margin=dict(l=10, r=10, t=40, b=10),
        height=chart_height,
        xaxis=dict(title='Energy', **grid_style()),
        yaxis=dict(title='Danceability', **grid_style()),
        showlegend=True,
        legend=dict(
            orientation='v', x=1.01, y=0.5,
            font=dict(size=9),
            bgcolor='rgba(17,17,24,0.8)',
            bordercolor='rgba(29,185,84,0.15)',
            borderwidth=1,
        ),
    )
    st.plotly_chart(fig_scatter2, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SECTION 5: GENRE BREAKDOWN + TOP ARTISTS
# ─────────────────────────────────────────────
section_header("🎼", "Genre & Artist Insights")

col_g1, col_g2 = st.columns([1, 1])

with col_g1:
    st.markdown('<div class="chart-card"><div class="chart-title">Top Genres by Avg Popularity</div><div class="chart-sub">20 genres ranked by mean track popularity score</div>', unsafe_allow_html=True)
    genre_pop = df_raw.groupby('track_genre')['popularity'].mean().nlargest(20).reset_index()
    genre_pop.columns = ['Genre', 'Avg Popularity']
    genre_pop = genre_pop.sort_values('Avg Popularity')
    colors_bar = [f'rgba(29,185,84,{0.4 + 0.6 * i / len(genre_pop)})' for i in range(len(genre_pop))]
    fig_genre = go.Figure(go.Bar(
        x=genre_pop['Avg Popularity'],
        y=genre_pop['Genre'],
        orientation='h',
        marker=dict(
            color=colors_bar,
            line=dict(width=0),
        ),
        hovertemplate='<b>%{y}</b><br>Avg Popularity: %{x:.1f}<extra></extra>',
        text=genre_pop['Avg Popularity'].round(1),
        textposition='outside',
        textfont=dict(size=10, color='#aaa'),
    ))
    fig_genre.update_layout(
        **PLOTLY_LAYOUT,
        height=520,
        xaxis=dict(title='Average Popularity', **grid_style(), range=[0, 80]),
        yaxis=dict(**grid_style()),
    )
    st.plotly_chart(fig_genre, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_g2:
    st.markdown('<div class="chart-card"><div class="chart-title">Audio Feature Distributions</div><div class="chart-sub">Violin plots showing spread per feature</div>', unsafe_allow_html=True)
    viol_feature = st.selectbox(
        "Select Feature",
        ['danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'instrumentalness', 'liveness'],
        key='viol_sel',
        label_visibility='collapsed',
    )
    top5_genres = df_raw.groupby('track_genre')['popularity'].mean().nlargest(6).index.tolist()
    df_viol = df_raw[df_raw['track_genre'].isin(top5_genres)]
    fig_viol = go.Figure()
    for j, genre in enumerate(top5_genres):
        vals = df_viol[df_viol['track_genre'] == genre][viol_feature].dropna()
        col_v = CLUSTER_COLORS[j % len(CLUSTER_COLORS)]
        fig_viol.add_trace(go.Violin(
            y=vals, name=genre,
            box_visible=True, meanline_visible=True,
            line_color=col_v,
            fillcolor=f'rgba({int(col_v[1:3],16)},{int(col_v[3:5],16)},{int(col_v[5:7],16)},0.15)',
            opacity=0.9,
            points=False,
        ))
    fig_viol.update_layout(
        **PLOTLY_LAYOUT,
        height=490,
        yaxis=dict(title=viol_feature.capitalize(), **grid_style()),
        xaxis=dict(**grid_style()),
        violingap=0.1,
        violingroupgap=0,
        showlegend=False,
    )
    st.plotly_chart(fig_viol, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SECTION 6: POPULARITY & TEMPO ANALYSIS
# ─────────────────────────────────────────────
section_header("📈", "Popularity & Tempo Deep-Dive")

col_p1, col_p2 = st.columns([1, 1])

with col_p1:
    st.markdown('<div class="chart-card"><div class="chart-title">Popularity Distribution</div><div class="chart-sub">How popular are the tracks in this dataset?</div>', unsafe_allow_html=True)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df_raw['popularity'],
        nbinsx=50,
        marker=dict(
            color='rgba(29,185,84,0.7)',
            line=dict(color='rgba(29,185,84,0.3)', width=0.5),
        ),
        hovertemplate='Popularity: %{x}<br>Count: %{y}<extra></extra>',
        name='All Tracks',
    ))
    fig_hist.update_layout(
        **PLOTLY_LAYOUT,
        height=360,
        xaxis=dict(title='Popularity Score', **grid_style()),
        yaxis=dict(title='Track Count', **grid_style()),
        bargap=0.05,
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_p2:
    st.markdown('<div class="chart-card"><div class="chart-title">Tempo vs Energy</div><div class="chart-sub">Hexbin density map — warmer = more tracks</div>', unsafe_allow_html=True)
    sample_te = df_raw.sample(n=min(15000, len(df_raw)), random_state=42)
    fig_hex = go.Figure(go.Histogram2dContour(
        x=sample_te['tempo'],
        y=sample_te['energy'],
        colorscale=[
            [0, 'rgba(0,0,0,0)'],
            [0.3, 'rgba(29,100,50,0.4)'],
            [0.6, 'rgba(29,185,84,0.6)'],
            [1, 'rgba(30,215,96,0.9)'],
        ],
        contours=dict(coloring='fill', showlines=False),
        ncontours=20,
        showscale=False,
        hovertemplate='Tempo: %{x:.0f} BPM<br>Energy: %{y:.2f}<extra></extra>',
    ))
    fig_hex.update_layout(
        **PLOTLY_LAYOUT,
        height=360,
        xaxis=dict(title='Tempo (BPM)', **grid_style()),
        yaxis=dict(title='Energy', **grid_style()),
    )
    st.plotly_chart(fig_hex, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SECTION 7: ELBOW CURVE (optional)
# ─────────────────────────────────────────────
if show_elbow:
    section_header("📐", "Optimal K — Elbow & Silhouette Analysis")
    with st.spinner("Computing elbow curve..."):
        k_vals, inertias, sils = compute_elbow(10, 6000)

    st.markdown('<div class="chart-card"><div class="chart-title">Elbow Curve + Silhouette Scores</div><div class="chart-sub">Find optimal number of clusters — elbow = best K tradeoff</div>', unsafe_allow_html=True)
    fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])
    fig_elbow.add_trace(go.Scatter(
        x=k_vals, y=inertias,
        name='Inertia (WCSS)',
        mode='lines+markers',
        line=dict(color='#1DB954', width=2.5),
        marker=dict(size=8, color='#1DB954', line=dict(color='white', width=1.5)),
        hovertemplate='K=%{x}<br>Inertia: %{y:,.0f}<extra></extra>',
    ), secondary_y=False)
    fig_elbow.add_trace(go.Scatter(
        x=k_vals, y=sils,
        name='Silhouette Score',
        mode='lines+markers',
        line=dict(color='#7c3aed', width=2.5, dash='dot'),
        marker=dict(size=8, color='#7c3aed', line=dict(color='white', width=1.5)),
        hovertemplate='K=%{x}<br>Silhouette: %{y:.3f}<extra></extra>',
    ), secondary_y=True)
    # Highlight selected K
    fig_elbow.add_vline(
        x=n_clusters, line_dash='dash',
        line_color='rgba(255,255,255,0.3)', line_width=1.5,
        annotation_text=f'  K={n_clusters}',
        annotation_font=dict(color='white', size=11),
    )
    fig_elbow.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#ffffff'),
        margin=dict(l=10, r=10, t=40, b=10),
        height=380,
        xaxis=dict(title='Number of Clusters (K)', **grid_style(), dtick=1),
        legend=dict(**LEGEND_BASE, orientation='h', x=0.5, y=-0.2, xanchor='center'),
    )
    fig_elbow.update_yaxes(title_text='Inertia (WCSS)', secondary_y=False, **grid_style())
    fig_elbow.update_yaxes(title_text='Silhouette Score', secondary_y=True, **grid_style())
    st.plotly_chart(fig_elbow, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SECTION 8: CLUSTER PROFILE TABLE
# ─────────────────────────────────────────────
section_header("📋", "Cluster Profiles")

st.markdown('<div class="chart-card"><div class="chart-title">Audio Feature Averages per Cluster</div><div class="chart-sub">Mean values of each audio feature — hover cells for details</div>', unsafe_allow_html=True)

profile_df = cluster_centers.copy()
profile_df.index = [cluster_names[i] if i < len(cluster_names) else f"Cluster {i}" for i in range(n_clusters)]
profile_df.columns = [c.capitalize() for c in profile_df.columns]

# Build plotly table
header_vals = ['<b>Cluster</b>'] + [f'<b>{c}</b>' for c in profile_df.columns]
cell_vals = [profile_df.index.tolist()] + [profile_df[c].round(3).tolist() for c in profile_df.columns]
cell_colors_rows = []
for i, col_name in enumerate(profile_df.columns):
    vals = profile_df[col_name].values
    vmin, vmax = vals.min(), vals.max()
    col_cells = []
    for v in vals:
        alpha = 0.1 + 0.5 * (v - vmin) / (vmax - vmin + 1e-9)
        col_cells.append(f'rgba(29,185,84,{alpha:.2f})')
    cell_colors_rows.append(col_cells)

cluster_col_colors = [hex_to_rgba(CLUSTER_COLORS[i % len(CLUSTER_COLORS)], 0.2) for i in range(n_clusters)]



fig_table = go.Figure(go.Table(
    columnwidth=[200] + [100] * len(profile_df.columns),
    header=dict(
        values=header_vals,
        fill_color=['#111118'] + ['#111118'] * len(profile_df.columns),
        font=dict(color=['#1DB954'] + ['#aaa'] * len(profile_df.columns), size=12),
        line_color='rgba(29,185,84,0.2)',
        align=['left'] + ['center'] * len(profile_df.columns),
        height=36,
    ),
    cells=dict(
        values=cell_vals,
        fill_color=[cluster_col_colors] + cell_colors_rows,
        font=dict(color='white', size=11),
        line_color='rgba(255,255,255,0.05)',
        align=['left'] + ['center'] * len(profile_df.columns),
        height=34,
    ),
))
fig_table.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color='#ffffff'),
    height=80 + 34 * n_clusters,
    margin=dict(l=0, r=0, t=0, b=0),
)
st.plotly_chart(fig_table, use_container_width=True, config={'displayModeBar': False})
st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SECTION 9: SPOTIFY GENRE RECOMMENDATIONS (WITH POSTERS & GIFS)
# ─────────────────────────────────────────────
section_header("✨", "Recommended Tracks Across Genres")

# Helper function to get base64 image data
def get_img_base64(img_path):
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

ASSETS = os.path.join(os.path.dirname(__file__), 'assets')
poster_files = {
    'synthwave': os.path.join(ASSETS, 'synthwave_poster_1785137523611.png'),
    'acoustic':  os.path.join(ASSETS, 'acoustic_poster_1785137539528.png'),
    'edm':       os.path.join(ASSETS, 'edm_poster_1785137554608.png'),
    'pop':       os.path.join(ASSETS, 'pop_poster_1785137566288.png'),
    'rock':      os.path.join(ASSETS, 'rock_poster_1785142799525.png'),
    'indie':     os.path.join(ASSETS, 'indie_poster_1785142818596.png'),
    'bollywood': os.path.join(ASSETS, 'bollywood_poster.png'),
    'bollywood2': os.path.join(ASSETS, 'bollywood_poster2.png'),
}

# Pre-encode local images to base64
b64_posters = {k: get_img_base64(v) for k, v in poster_files.items()}

# Curated High-Res Unsplash Poster maps for all genres (4 unique posters per genre)
GENRE_POSTER_MAP = {
    'pop': [
        f"data:image/png;base64,{b64_posters.get('pop', '')}",
        "https://images.unsplash.com/photo-1514525253161-7a46d19cd819?w=500&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4?w=500&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1470225620780-dba8ba36b745?w=500&auto=format&fit=crop&q=80",
    ],
    'acoustic': [
        f"data:image/png;base64,{b64_posters.get('acoustic', '')}",
        "https://images.unsplash.com/photo-1510915361894-db8b60106cb1?w=500&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1465847899084-d164df4dedc6?w=500&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1445375011782-4a2d8b6a5700?w=500&auto=format&fit=crop&q=80",
    ],
    'edm': [
        f"data:image/png;base64,{b64_posters.get('edm', '')}",
        "https://images.unsplash.com/photo-1516450360452-9312f5e86fc7?w=500&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1508700115892-45ecd05ae2ad?w=500&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1492684223066-81342ee5ff30?w=500&auto=format&fit=crop&q=80",
    ],
    'synthwave': [
        f"data:image/png;base64,{b64_posters.get('synthwave', '')}",
        "https://images.unsplash.com/photo-1509198397868-475647b2a1e5?w=500&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1518709268805-4e9042af9f23?w=500&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1550684848-fac1c5b4e853?w=500&auto=format&fit=crop&q=80",
    ],
    'rock': [
        f"data:image/png;base64,{b64_posters.get('rock', '')}",
        "https://images.unsplash.com/photo-1498038432885-c6f3f1b912ee?w=500&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1526478806334-5fd488fcaabc?w=500&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1511735111819-9a3f7709049c?w=500&auto=format&fit=crop&q=80",
    ],
    'indie': [
        f"data:image/png;base64,{b64_posters.get('indie', '')}",
        "https://images.unsplash.com/photo-1525994886773-080587e161c2?w=500&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1459749411175-04bf5292ceea?w=500&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1487180144351-b8472da7d491?w=500&auto=format&fit=crop&q=80",
    ],
    'classical': [
        "https://images.unsplash.com/photo-1520523839897-bd0b52f945a0?w=500&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1513829596324-4bb2800c5efb?w=500&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1507838153414-b4b713384a76?w=500&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1465847899084-d164df4dedc6?w=500&auto=format&fit=crop&q=80",
    ],
    'jazz': [
        "https://images.unsplash.com/photo-1511192336575-5a79af67a629?w=500&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1525994886773-080587e161c2?w=500&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1415201364774-f6f0bb35f28f?w=500&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1503095396549-807759245b35?w=500&auto=format&fit=crop&q=80",
    ],
    'indian': [
        f"data:image/png;base64,{b64_posters.get('bollywood', '')}",
        f"data:image/png;base64,{b64_posters.get('bollywood2', '')}",
        "https://images.unsplash.com/photo-1564769662533-4f00a87b4056?w=500&auto=format&fit=crop&q=80",
        "https://images.unsplash.com/photo-1598488035139-bdbb2231ce04?w=500&auto=format&fit=crop&q=80",
    ],
}

# GIF URLs for high energy ambient vibes
equalizer_gif = "https://media.giphy.com/media/l41K3o5TzManKz864/giphy.gif"
vinyl_gif = "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdWJ2dmtnM3lrdnlqdnlreXpybWs4cGpmZTRxZzFnZ3JvNWNxc3JzeSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3o85xoi6nNqVpBUwT6/giphy.gif"

st.markdown(f"""
<div style="background: linear-gradient(135deg, rgba(29,185,84,0.08) 0%, rgba(124,58,237,0.08) 100%);
            border: 1px solid rgba(29,185,84,0.2); border-radius: 20px; padding: 24px; margin-bottom: 25px;">
  <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:15px; margin-bottom:20px;">
    <div>
      <div style="display:flex; align-items:center; gap:10px;">
        <img src="{vinyl_gif}" style="width:32px; height:32px; border-radius:50%; object-fit:cover;">
        <h3 style="margin:0; font-size:1.4rem; color:#fff; font-weight:800;">Curated Spotify Radio Recommendations</h3>
      </div>
      <p style="margin:4px 0 0 0; color:#b3b3b3; font-size:0.88rem;">Handpicked top tracks grouped by genre with custom high-res album art</p>
    </div>
    <div style="display:flex; align-items:center; gap:8px; background:rgba(0,0,0,0.4); padding:6px 14px; border-radius:100px; border:1px solid rgba(29,185,84,0.3);">
      <img src="{equalizer_gif}" style="height:18px; width:24px; object-fit:cover; border-radius:2px;">
      <span style="font-size:0.78rem; color:#1DB954; font-weight:700; letter-spacing:0.05em;">LIVE AUDIO DNA MATCHING</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

rec_genres = ['pop', 'acoustic', 'edm', 'synthwave', 'rock', 'indie', 'classical', 'jazz', 'indian']
_genre_labels = {
    'pop': '🎤 Pop', 'acoustic': '🎸 Acoustic', 'edm': '🎛️ EDM',
    'synthwave': '🌆 Synthwave', 'rock': '🤘 Rock', 'indie': '🌿 Indie',
    'classical': '🎻 Classical', 'jazz': '🎺 Jazz', 'indian': '🪘 Bollywood / Indian',
}
rec_genre_selected = st.selectbox(
    "Select Genre for Targeted AI Recommendations",
    options=rec_genres,
    format_func=lambda g: _genre_labels.get(g, g.title()),
    index=0
)

target_g = rec_genre_selected.lower()  # already lower
df_rec_pool = df_raw[df_raw['track_genre'].str.contains(target_g, case=False, na=False)]

if len(df_rec_pool) < 4:
    df_rec_pool = df_raw.sample(n=100, random_state=42)

rec_sample = df_rec_pool.nlargest(4, 'popularity').reset_index(drop=True)

rec_cols = st.columns(4)

# Retrieve genre-specific posters
genre_posters = GENRE_POSTER_MAP.get(target_g, GENRE_POSTER_MAP['pop'])

for idx, col in enumerate(rec_cols):
    if idx < len(rec_sample):
        track = rec_sample.iloc[idx]
        img_src = genre_posters[idx % len(genre_posters)]

        with col:
            t_title = str(track['track_name']).replace('<', '&lt;').replace('>', '&gt;')
            t_artist = str(track['artists']).replace('<', '&lt;').replace('>', '&gt;')

            # ─── Card (no external Spotify link — play is in-dashboard) ───
            st.markdown(
                f'<div class="spotify-card">'
                f'<div class="spotify-poster-container">'
                f'<img src="{img_src}" class="spotify-poster-img" alt="Poster">'
                f'<div class="spotify-play-btn" style="cursor:default;">'
                f'<svg width="20" height="20" viewBox="0 0 24 24" fill="#000">'
                f'<polygon points="5 3 19 12 5 21 5 3"></polygon>'
                f'</svg>'
                f'</div>'
                f'</div>'
                f'<div class="spotify-card-title">{t_title}</div>'
                f'<div class="spotify-card-artist">{t_artist}</div>'
                f'<div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px; margin-bottom:6px;">'
                f'<span class="spotify-card-badge">{rec_genre_selected}</span>'
                f'<span style="font-size:0.75rem; color:#1DB954; font-weight:700;">★ {int(track["popularity"])}</span>'
                f'</div>'
                f'<div style="font-size:0.68rem; color:#888; margin-bottom:6px;">'
                f'🎵 Press ▶ below to preview · {int(float(track.get("tempo",120)))} BPM'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            # ─────────────────────────────────────────────────────────
            # GENRE-DISTINCT In-Dashboard Audio Synthesizer
            # Each genre has its own unique harmonic/rhythmic fingerprint
            # so every genre sounds genuinely different.
            # ─────────────────────────────────────────────────────────
            t_tempo  = max(60.0,  min(200.0, float(track.get('tempo', 120))))
            t_energy = max(0.01, float(track.get('energy', 0.5)))
            t_valence= max(0.01, float(track.get('valence', 0.5)))
            t_dance  = max(0.01, float(track.get('danceability', 0.5)))
            SR = 22050
            dur = 6.0
            t = np.linspace(0, dur, int(SR * dur), endpoint=False)
            bps = t_tempo / 60.0  # beats per second

            def _sin(f, amp=1.0): return np.sin(2 * np.pi * f * t) * amp
            def _saw(f, amp=1.0, n=8):
                """Sawtooth via Fourier series (n harmonics)."""
                s = np.zeros_like(t)
                for k in range(1, n + 1):
                    s += ((-1) ** (k + 1)) / k * np.sin(2 * np.pi * k * f * t)
                return s * (2 / np.pi) * amp
            def _sqr(f, amp=1.0, n=8):
                """Square wave via Fourier series (n odd harmonics)."""
                s = np.zeros_like(t)
                for k in range(1, n * 2, 2):
                    s += (1 / k) * np.sin(2 * np.pi * k * f * t)
                return s * (4 / np.pi) * amp
            def _beat_env(strength=1.0):
                return np.abs(np.sin(np.pi * bps * t)) ** (2.0 - strength)
            def _pluck(f, amp=1.0, decay=3.0):
                """Plucked string: sine with exponential decay."""
                return np.sin(2 * np.pi * f * t) * amp * np.exp(-decay * t)

            # Base root frequency from valence (C3→C4 range)
            root = 130.81 + t_valence * 130.0

            if target_g in ('edm', 'club', 'dance', 'dancehall', 'chicago-house',
                             'deep-house', 'detroit-techno', 'dubstep', 'drum-and-bass'):
                # ── EDM / House / Techno ──
                # Sawtooth bass + square lead + hard 4-on-floor kick
                kick_env = np.exp(-10.0 * (t % (1.0 / bps)))
                kick = np.sin(2 * np.pi * 55 * t) * kick_env * 0.5  # 808 sub kick
                bass = _sqr(root * 0.5, amp=0.3, n=6)
                lead = _saw(root * 2, amp=0.2 * t_energy, n=6)
                hat_noise = np.random.default_rng(42).standard_normal(len(t)) * 0.05
                hat_env   = np.abs(np.sin(np.pi * bps * 2 * t)) ** 4
                hat = hat_noise * hat_env
                wave  = (kick + bass + lead + hat) * _beat_env(t_dance)

            elif target_g in ('synthwave', 'vapor-death-pop', 'synth-pop', 'electro'):
                # ── Synthwave ──
                # Detuned twin sawtooth pads + slow arpeggio + retro square melody
                saw1 = _saw(root,        amp=0.22, n=10)
                saw2 = _saw(root * 1.005, amp=0.22, n=10)  # slight detune for chorus effect
                pad  = (saw1 + saw2) * 0.5
                arp_freqs = [root * 2, root * 2.5, root * 3, root * 4]
                arp = np.zeros_like(t)
                step = 1.0 / bps
                for ai, af in enumerate(arp_freqs):
                    mask = ((t >= ai * step) & (t < (ai + 1) * step))
                    arp[mask] = _sqr(af, amp=0.18, n=5)[mask]
                reverb_decay = np.exp(-1.5 * t)
                wave = (pad + arp) * reverb_decay * 0.8 + pad * 0.2

            elif target_g in ('rock', 'alt-rock', 'alternative', 'hard-rock',
                               'metal', 'black-metal', 'death-metal', 'heavy-metal',
                               'punk-rock', 'psych-rock', 'grunge'):
                # ── Rock / Metal ──
                # Power chord (root + fifth) in sawtooth, hard beat envelope
                root2  = root * 1.498  # perfect fifth (power chord)
                riff1  = _saw(root,  amp=0.30, n=12)
                riff2  = _saw(root2, amp=0.25, n=12)
                dist   = np.tanh((riff1 + riff2) * 3) * 0.55  # soft clipping = distortion
                snare_env = np.abs(np.sin(np.pi * bps * 2 * t + np.pi)) ** 6
                snare = np.random.default_rng(7).standard_normal(len(t)) * 0.12 * snare_env
                wave  = dist * _beat_env(t_energy) + snare

            elif target_g in ('acoustic', 'singer-songwriter', 'folk',
                               'country', 'bluegrass'):
                # ── Acoustic / Folk ──
                # Gentle plucked guitar chord (root + maj3rd + fifth) with soft decay
                maj3  = root * 1.260
                fifth = root * 1.498
                p1 = _pluck(root,  amp=0.40, decay=1.5)
                p2 = _pluck(maj3,  amp=0.28, decay=2.0)
                p3 = _pluck(fifth, amp=0.20, decay=2.5)
                p_high = _pluck(root * 2, amp=0.15, decay=3.0)
                strum_period = 60.0 / t_tempo * 2
                strum_env = np.abs(np.sin(np.pi / strum_period * t)) ** 0.5
                wave = (p1 + p2 + p3 + p_high) * strum_env

            elif target_g in ('classical', 'opera', 'piano', 'symphony'):
                # ── Classical / Orchestral ──
                # Pure sine chords with slow vibrato and no beat pulse (legato)
                vibrato = 1.0 + 0.005 * np.sin(2 * np.pi * 5.5 * t)  # 5.5 Hz vibrato
                root_v  = root  * vibrato
                third_v = root * 1.260 * vibrato
                fifth_v = root * 1.498 * vibrato
                wave  = _sin(root_v,  amp=0.38)
                wave += _sin(third_v, amp=0.24)
                wave += _sin(fifth_v, amp=0.20)
                wave += _sin(root * 2 * vibrato, amp=0.12)
                # Smooth ADSR (attack 0.8s, sustain, release 1.2s)
                adsr = np.ones(len(t))
                att = int(SR * 0.8); rel = int(SR * 1.2)
                adsr[:att]  = np.linspace(0, 1, att)
                adsr[-rel:] = np.linspace(1, 0, rel)
                wave *= adsr

            elif target_g in ('jazz', 'blues', 'soul', 'funk', 'bossanova', 'samba'):
                # ── Jazz / Blues ──
                # Tritone substitute + flat-7 blue note + swung shuffle rhythm
                flat7 = root * 1.782  # minor 7th
                trit  = root * 1.414  # tritone (#4 blue note)
                wave  = _sin(root,  amp=0.30)
                wave += _sin(flat7, amp=0.20)
                wave += _sin(trit,  amp=0.14 * t_valence)  # blue note intensity
                wave += _sin(root * 2, amp=0.10)
                # Swing: off-beat accent
                swing = np.abs(np.sin(np.pi * bps * 1.5 * t + np.pi * 0.25)) ** 2
                wave *= (0.4 + 0.6 * swing)

            elif target_g in ('indie', 'indie-pop', 'garage', 'emo', 'j-rock'):
                # ── Indie ──
                # Jangly major 7th chord (sine + slight saw texture) + lo-fi noise floor
                maj7  = root * 1.875  # major 7th
                third = root * 1.260
                fifth = root * 1.498
                wave  = _sin(root,  amp=0.28)
                wave += _sin(third, amp=0.20)
                wave += _sin(fifth, amp=0.18)
                wave += _sin(maj7,  amp=0.12)
                wave += _saw(root, amp=0.08, n=4)  # light jangle texture
                lo_fi = np.random.default_rng(13).standard_normal(len(t)) * 0.02
                wave  = wave * _beat_env(t_dance * 0.7) + lo_fi

            elif target_g in ('indian', 'bollywood', 'desi', 'bhangra'):
                # ── Bollywood / Indian Classical ──
                # Sa-Re-Ga-Ma pentatonic scale intervals with tabla-style beat + microtone shimmer
                # Raga intervals: Sa(1) Re(9/8) Ga(5/4) Pa(3/2) Ni(15/8)
                sa  = root
                re  = root * 1.125   # 9/8
                ga  = root * 1.25    # 5/4
                pa  = root * 1.5     # 3/2 (Pa)
                ni  = root * 1.875   # 15/8 (Ni)
                # Shimmer: microtone oscillation (gamak ornament)
                shimmer = 1.0 + 0.008 * np.sin(2 * np.pi * 7 * t)
                wave  = _sin(sa * shimmer, amp=0.32)
                wave += _sin(re * shimmer, amp=0.18)
                wave += _sin(ga,           amp=0.16)
                wave += _sin(pa * shimmer, amp=0.20)
                wave += _sin(ni,           amp=0.10 * t_valence)
                wave += _sin(sa * 2,       amp=0.12)  # taar saptak (upper octave)
                # Tabla-style beat: 16-beat teentaal approximation
                tabla_period = 60.0 / t_tempo
                bol_env = (np.abs(np.sin(np.pi / tabla_period * t)) ** 3 +
                           0.5 * np.abs(np.sin(np.pi / tabla_period * 2 * t)) ** 5)
                wave *= bol_env

            else:
                # ── Pop (default) ──
                # Bright major chord, punchy beat, melodic hook
                third = root * 1.260  # major third
                fifth = root * 1.498
                wave  = _sin(root,      amp=0.32)
                wave += _sin(third,     amp=0.24)
                wave += _sin(fifth,     amp=0.20)
                wave += _sin(root * 2,  amp=0.16 * t_energy)
                hook  = _sin(root * 2.5, amp=0.12 * t_energy)
                hook_env = np.abs(np.sin(np.pi * bps * t * 0.5)) ** 1.2
                wave += hook * hook_env
                wave *= _beat_env(t_dance)

            # Universal finish: fade-in (0.35s) + fade-out (0.9s) + normalize
            fi = int(SR * 0.35); fo = int(SR * 0.9)
            wave[:fi]  *= np.linspace(0.0, 1.0, fi)
            wave[-fo:] *= np.linspace(1.0, 0.0, fo)
            peak = np.max(np.abs(wave))
            if peak > 0:
                wave = (wave / peak * 0.88).astype(np.float32)

            st.audio(wave, sample_rate=SR)

st.markdown('<br>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SECTION 10: TOP TRACKS PER CLUSTER
# ─────────────────────────────────────────────
section_header("🏆", "Top Tracks per Cluster")

tabs = st.tabs([
    cluster_names[i] if i < len(cluster_names) else f"Cluster {i}"
    for i in range(n_clusters)
])
for i, tab in enumerate(tabs):
    with tab:
        top_tracks = clustered[clustered['cluster'] == i].nlargest(8, 'popularity')[
            ['track_name', 'artists', 'popularity', 'energy', 'danceability', 'valence', 'track_genre']
        ].reset_index(drop=True)

        color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
        rows_html = ""
        for idx, row in top_tracks.iterrows():
            bar_w = int(row['popularity'])
            t_name = str(row['track_name'])[:45].replace('<', '&lt;').replace('>', '&gt;')
            a_name = str(row['artists'])[:40].replace('<', '&lt;').replace('>', '&gt;')
            g_name = str(row['track_genre']).replace('<', '&lt;').replace('>', '&gt;')
            rows_html += (
                f'<div class="song-row">'
                f'<span class="song-num">{idx+1}</span>'
                f'<div style="flex:1;">'
                f'<div class="song-name">{t_name}</div>'
                f'<div class="song-artist">{a_name} &nbsp;·&nbsp; <span class="tag-chip">{g_name}</span></div>'
                f'</div>'
                f'<div style="text-align:right; min-width:100px;">'
                f'<div style="font-size:0.8rem; color:{color}; font-weight:700; margin-bottom:4px;">{int(row["popularity"])}</div>'
                f'<div class="progress-bar-bg" style="width:80px; display:inline-block;">'
                f'<div class="progress-bar-fill" style="width:{bar_w}%; background:{color};"></div>'
                f'</div>'
                f'</div>'
                f'</div>'
            )
        st.markdown(
            f'<div class="chart-card" style="border-left: 3px solid {color};">'
            f'<div class="chart-title" style="color:{color}; margin-bottom:12px;">'
            f'🎵 {cluster_names[i] if i < len(cluster_names) else f"Cluster {i}"} — Top Tracks'
            f'</div>'
            f'{rows_html}'
            f'</div>',
            unsafe_allow_html=True
        )

        # Mini feature comparison for this cluster
        c_data = clustered[clustered['cluster'] == i]
        mini_cols = st.columns(4)
        mini_metrics = [
            ('⚡ Avg Energy', f"{c_data['energy'].mean():.2f}", color),
            ('💃 Avg Dance', f"{c_data['danceability'].mean():.2f}", '#7c3aed'),
            ('😊 Avg Valence', f"{c_data['valence'].mean():.2f}", '#0ea5e9'),
            ('🌟 Avg Popularity', f"{c_data['popularity'].mean():.1f}", '#f97316'),
        ]
        for mc, (label, val, mc_color) in zip(mini_cols, mini_metrics):
            mc.markdown(f"""
            <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                        border-radius:12px; padding:14px; text-align:center; margin-top:12px;">
              <div style="font-size:1.4rem; font-weight:800; color:{mc_color};">{val}</div>
              <div style="font-size:0.72rem; color:#888; margin-top:4px;">{label}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SECTION 10: FEATURE IMPORTANCE BAR
# ─────────────────────────────────────────────
section_header("🎛️", "Feature Importance in Clustering")

st.markdown('<div class="chart-card"><div class="chart-title">PCA Component Loadings</div><div class="chart-sub">Which audio features drive the most variance?</div>', unsafe_allow_html=True)

scaler_imp = StandardScaler()
X_imp = scaler_imp.fit_transform(df_raw.sample(n=min(20000, len(df_raw)), random_state=42)[FEATURES_FOR_CLUSTER])
pca_imp = PCA(n_components=3, random_state=42)
pca_imp.fit(X_imp)

loadings = pd.DataFrame(
    pca_imp.components_.T,
    columns=[f'PC{i+1} ({pca_variance[i]*100:.1f}%)' for i in range(3)],
    index=[f.capitalize() for f in FEATURES_FOR_CLUSTER],
)

fig_loading = go.Figure()
colors_load = ['#1DB954', '#7c3aed', '#0ea5e9']
for j, col in enumerate(loadings.columns):
    fig_loading.add_trace(go.Bar(
        name=col,
        x=loadings.index.tolist(),
        y=loadings[col].tolist(),
        marker_color=colors_load[j],
        opacity=0.85,
        hovertemplate=f'<b>%{{x}}</b><br>{col}: %{{y:.3f}}<extra></extra>',
    ))
fig_loading.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color='#ffffff'),
    margin=dict(l=10, r=10, t=40, b=10),
    height=380,
    barmode='group',
    bargap=0.15,
    xaxis=dict(**grid_style()),
    yaxis=dict(title='Loading Coefficient', **grid_style(), zeroline=True),
    legend=dict(**LEGEND_BASE, orientation='h', x=0.5, y=1.1, xanchor='center'),
)
st.plotly_chart(fig_loading, use_container_width=True, config={'displayModeBar': False})
st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SECTION 11: EXPLICIT vs CLEAN
# ─────────────────────────────────────────────
section_header("🎭", "Explicit vs Clean — Feature Comparison")

col_ex1, col_ex2 = st.columns(2)

with col_ex1:
    st.markdown('<div class="chart-card"><div class="chart-title">Content Distribution</div><div class="chart-sub">Explicit vs non-explicit tracks</div>', unsafe_allow_html=True)
    expl_counts = df_raw['explicit'].value_counts().reset_index()
    expl_counts.columns = ['explicit', 'count']
    expl_counts['label'] = expl_counts['explicit'].map({True: '🔞 Explicit', False: '✅ Clean'})
    fig_expl = go.Figure(go.Pie(
        labels=expl_counts['label'],
        values=expl_counts['count'],
        hole=0.55,
        marker=dict(colors=['#f97316', '#1DB954'], line=dict(color='#0A0A0F', width=3)),
        textfont=dict(size=12, color='white'),
    ))
    fig_expl.update_layout(**PLOTLY_LAYOUT, height=300)
    st.plotly_chart(fig_expl, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_ex2:
    st.markdown('<div class="chart-card"><div class="chart-title">Energy & Danceability: Explicit vs Clean</div><div class="chart-sub">Grouped box comparison</div>', unsafe_allow_html=True)
    df_raw['Content'] = df_raw['explicit'].map({True: '🔞 Explicit', False: '✅ Clean'})
    fig_box2 = go.Figure()
    for feat, col_b in [('energy', '#1DB954'), ('danceability', '#7c3aed')]:
        for label, line_c in [('🔞 Explicit', '#f97316'), ('✅ Clean', '#60a5fa')]:
            vals = df_raw[df_raw['Content'] == label][feat]
            fig_box2.add_trace(go.Box(
                y=vals.sample(n=min(5000, len(vals)), random_state=42),
                name=f"{feat.capitalize()} ({label})",
                marker_color=line_c if 'Explicit' in label else col_b,
                line=dict(color=line_c if 'Explicit' in label else col_b),
                boxmean='sd',
                fillcolor=f'rgba({int(col_b[1:3],16) if "Clean" in label else 249},{int(col_b[3:5],16) if "Clean" in label else 115},{int(col_b[5:7],16) if "Clean" in label else 22},0.15)',
            ))
    fig_box2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter, sans-serif', color='#ffffff'),
        margin=dict(l=10, r=10, t=40, b=10),
        height=300,
        showlegend=True,
        boxmode='group',
        yaxis=dict(**grid_style()),
        xaxis=dict(**grid_style()),
        legend=dict(
            bgcolor='rgba(17,17,24,0.8)',
            bordercolor='rgba(29,185,84,0.2)',
            borderwidth=1,
            font=dict(size=9)
        ),
    )
    st.plotly_chart(fig_box2, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center; padding:30px 0 20px; color:#555; font-size:0.82rem;">
  <div style="font-size:1.5rem; margin-bottom:10px;">🎵</div>
  <div style="color:#888; font-weight:500;">
    SpotiCluster Dashboard &nbsp;·&nbsp; Built with Streamlit + Plotly
  </div>
  <div style="margin-top:6px; color:#555;">
    {len(df_raw):,} tracks &nbsp;·&nbsp; {df_raw['track_genre'].nunique()} genres &nbsp;·&nbsp;
    K-Means with K={n_clusters} &nbsp;·&nbsp;
    Silhouette: {sil_score:.3f}
  </div>
</div>
""", unsafe_allow_html=True)
