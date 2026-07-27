# 🎵 Spotify Songs Clustering Dashboard

An interactive ML-powered music analytics dashboard built with **Streamlit** and **Plotly**, featuring K-Means clustering on Spotify audio features.

![Spotify Clustering](https://images.unsplash.com/photo-1614680376593-902f74cf0d41?w=800&auto=format&fit=crop&q=80)

## ✨ Features

- **K-Means Clustering** — Group 100K+ tracks by audio DNA (danceability, energy, valence, tempo, etc.)
- **PCA Visualization** — 2D & 3D principal component scatter plots
- **Interactive Radar Charts** — Per-cluster audio feature profiles
- **Elbow & Silhouette Analysis** — Find the optimal number of clusters
- **Genre Recommendations** — Handpicked top tracks per genre with album art
- **In-Dashboard Audio Preview** — Genre-distinct synthesized previews (EDM, Rock, Jazz, Bollywood, Synthwave & more)
- **Bollywood / Indian Music** — Dedicated Bollywood section with raga-based audio synthesis
- **Spotify-inspired UI** — Dark theme, animated waveforms, glassmorphism cards

## 🎛️ Genre Audio Synthesizer

Each genre produces a **distinctly different** synthesized preview:

| Genre | Audio Character |
|---|---|
| 🎤 Pop | Bright major chord + punchy beat |
| 🎸 Acoustic | Plucked guitar with strum envelope |
| 🎛️ EDM | 808 sub kick + sawtooth bass + hi-hats |
| 🌆 Synthwave | Detuned twin-saw pads + arpeggio |
| 🤘 Rock | Distorted power chord (tanh clipping) |
| 🎺 Jazz | Blue notes + swing shuffle rhythm |
| 🎻 Classical | Vibrato legato with ADSR |
| 🪘 Bollywood | Raga pentatonic + tabla-style teentaal beat |

## 🚀 Getting Started

### Prerequisites

```bash
pip install streamlit pandas numpy plotly scikit-learn
```

### Run the Dashboard

```bash
streamlit run dashboard.py
```

Then open **http://localhost:8501** in your browser.

## 📊 Dataset

The project uses the [Spotify Tracks Dataset](https://www.kaggle.com/) with **114 genres** and **100K+ tracks**, each with 9 audio features:
- `danceability`, `energy`, `loudness`, `speechiness`
- `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Streamlit** | Web app framework |
| **Plotly** | Interactive charts |
| **scikit-learn** | K-Means, PCA, Silhouette |
| **NumPy** | Audio synthesis |
| **Pandas** | Data wrangling |

## 📁 Project Structure

```
├── dashboard.py          # Main Streamlit app
├── dataset.csv           # Spotify tracks dataset
├── assets/               # Generated album art posters
├── .streamlit/           # Streamlit config
└── README.md
```

## 📸 Dashboard Sections

1. **Hero Banner** — Live cluster stats and animated waveform
2. **KPI Metrics** — Track count, cluster count, silhouette score, avg energy & danceability
3. **Cluster Scatter** — PCA 2D projection of all clusters
4. **Radar Charts** — Per-cluster audio fingerprints
5. **3D Cluster Plot** — Interactive 3D PCA view
6. **Elbow Analysis** — Inertia + silhouette vs K
7. **Genre Recommendations** — Top tracks with album art + in-dashboard audio
8. **Top Tracks per Cluster** — Popularity-ranked track lists
9. **Feature Importance** — PCA loading analysis
10. **Explicit vs Clean** — Content type comparison

---

Built with ❤️ using Streamlit + Plotly + scikit-learn
