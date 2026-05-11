import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Generate Dataset
X, y = make_blobs(
    n_samples=300,
    centers=3,
    n_features=2,
    cluster_std=1.2,
    random_state=42
)

df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
print(df.head())

# Select Model
model = KMeans(n_clusters=3, random_state=42)

# model = GaussianMixture(n_components=3, random_state=42)

# Apply Model
if isinstance(model, KMeans):
    model.fit(X)
    labels = model.labels_
    centers = model.cluster_centers_

else:
    model.fit(X)
    labels = model.predict(X)
    centers = model.means_

# Print Results
print("Cluster Labels:")
print(labels)

print("\nCluster Centers:")
print(centers)

# Silhouette Score
score = silhouette_score(X, labels)
print("\nSilhouette Score :", score)

# Plot Clustering Graph
plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', s=50)

plt.scatter(
    centers[:,0],
    centers[:,1],
    color='red',
    marker='X',
    s=200,
    label='Centers'
)

plt.title("Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()


# ==========================
# Elbow Method (for KMeans)
# ==========================

wcss = []

for i in range(1,11):
    km = KMeans(n_clusters=i, random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)

plt.figure(figsize=(8,6))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()