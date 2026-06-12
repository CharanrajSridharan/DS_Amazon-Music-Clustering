# 🎵 Amazon Music Clustering

> Automatically grouping 95,837 songs into meaningful clusters using unsupervised machine learning — without any genre labels.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-green?logo=pandas)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Project Overview

With millions of songs on Amazon Music, manually categorizing tracks is impractical. This project uses **unsupervised machine learning** to automatically group songs by how they *sound* — analyzing audio features like energy, danceability, tempo, and speechiness.

**Three clustering algorithms were compared:**
- ✅ K-Means ← Winner
- 🔍 DBSCAN (outlier detection)
- 🌳 Hierarchical Clustering

**Result:** 3 distinct music clusters discovered — Spoken Word, Soft/Acoustic, and Energetic — without using any genre labels.

---

## 📁 Project Structure

```
amazon-music-clustering/
│
├── 📓 amazon_music_clustering.ipynb   ← Main notebook
├── 📄 single_genre_artists.csv        ← Input dataset
├── 📄 amazon_music_clustered.csv      ← Output with cluster labels
│
├── 📊 plots/
│   ├── feature_distributions.png      ← EDA distributions
│   ├── elbow_silhouette.png           ← K selection plots
│   ├── pca_clusters.png               ← 2D cluster visualization
│   ├── cluster_heatmap.png            ← Feature heatmap
│   ├── cluster_barcharts.png          ← Feature bar charts
│   └── dendrogram.png                 ← Hierarchical tree
│
├── 📝 Project_Report.md               ← Full documentation
└── README.md                          ← This file
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| File | single_genre_artists.csv |
| Songs | 95,837 |
| Original Columns | 23 |
| Features Used for ML | 10 |
| Missing Values | 0 |
| Duplicates | 0 |

**Audio features used:**
`danceability` • `energy` • `loudness` • `speechiness` • `acousticness` • `instrumentalness` • `liveness` • `valence` • `tempo` • `duration_ms`

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Run the project

```bash
# Clone the repo
git clone https://github.com/yourusername/amazon-music-clustering.git
cd amazon-music-clustering

# Launch Jupyter
jupyter notebook amazon_music_clustering.ipynb
```

---

## 🚀 Approach

```
📥 Load Data          →  95,837 songs × 23 features
🔍 Explore            →  Check dtypes, missing values, distributions
🧹 Clean              →  Drop text columns, verify no duplicates
⚖️  Normalize          →  StandardScaler (mean=0, std=1)
📊 Find k             →  Elbow Method + Silhouette Score → k=3
🤖 K-Means            →  3 clusters on full dataset
🔍 DBSCAN             →  Outlier detection (1,593 anomalies found)
🌳 Hierarchical       →  Confirms 3 clusters (10k sample)
📈 Evaluate           →  Silhouette, Davies-Bouldin, Inertia
🎨 Visualize          →  PCA 2D plot, heatmap, bar charts
💾 Export             →  CSV with cluster labels added
```

---

## 🎵 Results

### Cluster Summary

| Cluster | Name | Songs | Key Feature |
|---------|------|-------|-------------|
| 0 | 🎙️ Spoken Word | 12,513 (13%) | Speechiness: 0.83 |
| 1 | 🎸 Soft / Acoustic | 30,807 (32%) | Acousticness: 0.75 |
| 2 | 🎉 Energetic | 52,517 (55%) | Energy: 0.69, Tempo: 125 BPM |

### Cluster Profiles

**🎙️ Cluster 0 — Spoken Word / Audiobooks**
- Very high speechiness (0.83) — dominated by spoken words
- Shortest average duration (~1.6 minutes)
- Includes: audio dramas, podcasts, children's stories, spoken word

**🎸 Cluster 1 — Soft / Acoustic**
- Very high acousticness (0.75) — organic instruments
- Low energy (0.31) — calm and gentle
- Includes: ballads, folk, classical, soft pop

**🎉 Cluster 2 — Energetic / World Music**
- High energy (0.69) and loudness (-7.6 dB)
- Fastest tempo (124.9 BPM)
- Includes: pop, dance, Latin, electronic

### Evaluation Metrics

| Metric | Value | Verdict |
|--------|-------|---------|
| Silhouette Score | 0.2423 | Acceptable ✅ |
| Davies-Bouldin Index | 1.5702 | Acceptable ✅ |
| Inertia | 658,335 | Compact ✅ |

> **Note:** Moderate scores are expected for music data. Songs naturally blend across genres — a sad acoustic rap can belong to either spoken word or acoustic cluster. These scores are scientifically valid.

---

## 📈 Key Visualizations

### PCA Cluster Plot
Songs reduced to 2D using PCA (45.9% variance captured). Red = Spoken Word, Blue = Acoustic, Green = Energetic.

### Feature Heatmap
Average audio features per cluster — green = high value, red = low value.

### Elbow + Silhouette
Both methods independently confirm k=3 as optimal.

---

## 💡 Key Insights

1. **Speechiness** is the single strongest separator between spoken word and music
2. **Energy + Loudness** cleanly divide the two music clusters
3. **Audio features work across all languages** — songs in Japanese, German, French, Hebrew all correctly grouped
4. **1,593 genuine outliers** detected by DBSCAN — includes 13-minute recordings and 1920s audio
5. **All 3 clustering methods confirmed 3 natural groups** in the data

---

## 🏗️ Skills Demonstrated

| Skill | Applied In |
|-------|-----------|
| Data Exploration (EDA) | Shape, dtypes, null check, distributions |
| Data Cleaning | Duplicate check, column dropping |
| Feature Selection | Keeping only audio features |
| Data Normalization | StandardScaler |
| K-Means Clustering | Main clustering algorithm |
| Elbow Method | Finding optimal k |
| Silhouette Score | Cluster quality evaluation |
| DBSCAN | Outlier/anomaly detection |
| Hierarchical Clustering | Dendrogram visualization |
| PCA | Dimensionality reduction for visualization |
| Heatmaps & Bar Charts | Cluster interpretation |
| Data Storytelling | Cluster naming and interpretation |

---

## 📚 Technologies Used

```python
import pandas as pd              # Data manipulation
import numpy as np               # Numerical operations
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns            # Statistical visualization
from sklearn.preprocessing import StandardScaler      # Normalization
from sklearn.cluster import KMeans, DBSCAN           # Clustering
from sklearn.cluster import AgglomerativeClustering  # Hierarchical
from sklearn.metrics import silhouette_score         # Evaluation
from sklearn.metrics import davies_bouldin_score     # Evaluation
from sklearn.decomposition import PCA                # Visualization
from scipy.cluster.hierarchy import dendrogram       # Dendrogram
```

---

## 🤝 Acknowledgements

- **GUVI & HCL** — Project specification and guidance
- **Spotify Audio Features** — Dataset feature definitions reference
- **scikit-learn documentation** — Algorithm implementations

---

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

---
