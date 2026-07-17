# Spotify Songs Clustering

This project applies unsupervised machine learning techniques to group Spotify songs based on their audio features. The goal is to discover natural patterns in music and understand how songs can be categorized without using predefined labels.

Instead of predicting an output, clustering identifies songs that share similar characteristics such as energy, danceability, acousticness, tempo, loudness, and other audio features.

---

## Project Overview

Music streaming platforms contain millions of songs with different styles and characteristics. Organizing these songs manually is difficult, so clustering algorithms can be used to automatically group similar tracks.

In this project, the Spotify Songs dataset is explored, cleaned, visualized, and clustered using multiple machine learning algorithms. The resulting clusters can support music recommendation systems, playlist generation, and song similarity analysis.

---
-
## Dataset

The dataset contains information about Spotify songs along with several audio features extracted from each track.

Some of the important features include:

- Popularity
- Danceability
- Energy
- Loudness
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo
- Duration
- Explicit Content
- Track Genre

Since this is an unsupervised learning problem, the dataset does not contain a target variable.

---

## Project Workflow

- Data Loading
- Data Cleaning
- Missing Value Handling
- Duplicate Removal
- Exploratory Data Analysis (EDA)
- Feature Selection
- Feature Scaling
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN Clustering
- Elbow Method
- Silhouette Score Evaluation
- PCA for Cluster Visualization
- Cluster Analysis and Interpretation

---

## Exploratory Data Analysis

Several visualizations were created to better understand the dataset, including:

- Distribution of numerical features
- Genre analysis
- Artist analysis
- Popularity distribution
- Correlation heatmap
- Boxplots
- Word Clouds
- Scatter plots
- Pair plots
- Feature relationship analysis

These visualizations helped identify trends, feature relationships, and potential patterns before applying clustering algorithms.

---

## Machine Learning Algorithms

The following clustering algorithms were implemented and compared:

- K-Means Clustering
- Agglomerative Hierarchical Clustering
- DBSCAN

The performance of each algorithm was evaluated using the Silhouette Score, and the most suitable clustering approach was selected based on the results.

---

## Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- WordCloud

---

## Results

The clustering models successfully grouped songs with similar audio characteristics into distinct clusters.

The analysis showed that songs with similar energy, danceability, tempo, acousticness, and valence naturally formed meaningful groups. These clusters can be useful for building recommendation systems and generating personalized playlists.

---

## Future Improvements

Some possible improvements include:

- Testing additional clustering algorithms
- Hyperparameter optimization
- Interactive dashboards using Streamlit
- Genre-wise cluster analysis
- Song recommendation engine based on cluster similarity

---

## Repository Structure

```
Spotify-Songs-Clustering/
│
├── Spotify_Songs_Clustering.ipynb
├── dataset.csv
├── README.md
└── images/
```

---

## Author

**Dushyant Sharma**

If you found this project useful, feel free to star the repository.
