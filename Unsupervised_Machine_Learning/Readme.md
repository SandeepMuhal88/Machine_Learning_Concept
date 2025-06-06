# Unsupervised Learning

## 1. Detailed Explanation
Unsupervised learning is a type of machine learning where the algorithm is trained on **unlabeled data**. Unlike supervised learning, where the goal is to map input to a known output, unsupervised learning aims to discover the underlying structure or distribution in the data. The algorithm explores the data and tries to find patterns, clusters, or relationships without explicit guidance.

## Key Goals of Unsupervised Learning
- **Clustering** – Grouping data points with similar properties.
- **Dimensionality Reduction** – Reducing the number of features while preserving important information.
- **Anomaly Detection** – Identifying outliers or unusual data points.
- **Density Estimation** – Estimating the probability distribution of data.

---

## Common Unsupervised Learning Algorithms

### 1. Clustering Algorithms
Clustering aims to group similar data points into clusters based on some similarity measure.

**Examples:**
- **K-Means** – Divides data into k clusters based on the Euclidean distance from the cluster center.
- **Hierarchical Clustering** – Builds a hierarchy of clusters using either a bottom-up or top-down approach.
- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** – Groups data points that are closely packed together while identifying outliers.

---

### 2. Dimensionality Reduction Algorithms
These algorithms reduce the number of features while retaining the most important information.

**Examples:**
- **PCA (Principal Component Analysis)** – Projects data onto a lower-dimensional subspace using eigenvectors.
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)** – Preserves the local structure of the data for visualization.
- **Autoencoders** – Neural networks that reduce dimensions by learning an encoded representation.

---

### 3. Anomaly Detection
Identifying outliers or rare events that deviate significantly from the norm.

**Examples:**
- **Isolation Forest** – Isolates points by recursively splitting the data.
- **One-Class SVM** – Learns a boundary around the data to separate normal and abnormal instances.

---

### 4. Association Rule Learning
Discovering interesting relationships between variables in large datasets.

**Examples:**
- **Apriori Algorithm** – Identifies frequent item sets and derives association rules.
- **Eclat Algorithm** – Uses a depth-first search strategy to mine association rules.

---

