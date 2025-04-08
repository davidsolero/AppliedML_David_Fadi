# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# Load the dataset
file_path = 'Customer Data.csv'  # Adjust the path to your dataset
data = pd.read_csv(file_path)

# Step 1: Data Preprocessing
# Fill missing values for numeric columns
numeric_data = data.select_dtypes(include='number')
data[numeric_data.columns] = numeric_data.fillna(numeric_data.mean())

# Drop irrelevant columns
data = data.drop(columns=['CUST_ID'])

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

# Standardize the data
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Hierarchical Clustering
# Apply Hierarchical Clustering
hierarchical = linkage(data, method='complete')
plt.figure(figsize=(10, 7))
dendrogram(hierarchical)
plt.title("Dendrogram for Hierarchical Clustering")
plt.show()

# Define clusters for hierarchical clustering
hierarchical_labels = fcluster(hierarchical, t=3, criterion='maxclust')  # Adjust `t` as needed
data['Hierarchical_Cluster'] = hierarchical_labels

# Evaluate Hierarchical Clustering with silhouette score
hierarchical_silhouette = silhouette_score(data, hierarchical_labels)
print(f"Silhouette Score for Hierarchical Clustering: {hierarchical_silhouette:.2f}")

# DBSCAN Clustering
# Apply DBSCAN
dbscan = DBSCAN(eps=5, min_samples=10)
clusters_dbscan = dbscan.fit_predict(data_scaled)

# Add cluster labels to the dataset
data['Cluster'] = clusters_dbscan

# Evaluate DBSCAN with silhouette score
if len(set(clusters_dbscan)) > 1:  # Avoid single cluster scenario
    sil_score = silhouette_score(data_scaled, clusters_dbscan)
    print(f"Silhouette Score for DBSCAN: {sil_score}")
else:
    print("DBSCAN produced a single cluster.")

# Visualize Hierarchical Clusters with PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data)
data['PCA1'] = pca_result[:, 0]
data['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Hierarchical_Cluster', data=data, palette='viridis')
plt.title("Hierarchical Clusters Visualization (PCA)")
plt.show()

# Visualize Hierarchical Clusters with t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(data)
data['TSNE1'] = tsne_result[:, 0]
data['TSNE2'] = tsne_result[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='Hierarchical_Cluster', data=data, palette='tab10')
plt.title("Hierarchical Clusters Visualization (t-SNE)")
plt.show()

# Recommendations based on Hierarchical Clusters
print("Recommendations Based on Hierarchical Clusters:")
for cluster in data['Hierarchical_Cluster'].unique():
    cluster_data = data[data['Hierarchical_Cluster'] == cluster]
    details = cluster_data.mean()

    print(f"\nCluster {cluster}:")
    if details['CASH_ADVANCE'] > details['PURCHASES']:
        print("Recommendation: Offer short-term loans or debt consolidation plans.")
    elif details['PRC_FULL_PAYMENT'] > 0.8:
        print("Recommendation: Suggest premium wealth management services.")
    elif details['PURCHASES_FREQUENCY'] > 0.5:
        print("Recommendation: Introduce cashback or rewards programs.")
    else:
        print("Recommendation: Focus on improving customer engagement with personalized offers.")

# Dimensionality Reduction for Visualization
# PCA Visualization
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(10, 7))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters_dbscan, cmap='viridis', s=10)
plt.title("DBSCAN Clusters Visualized with PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()

# t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(data_scaled)

plt.figure(figsize=(10, 7))
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=clusters_dbscan, cmap='viridis', s=10)
plt.title("DBSCAN Clusters Visualized with t-SNE")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.colorbar(label="Cluster")
plt.show()

# Analyzing Clusters and Recommendations
cluster_summary = data.groupby('Cluster').mean()

print("Cluster Recommendations:")
for cluster, details in cluster_summary.iterrows():
    print(f"\nCluster {cluster}:")
    if details['CASH_ADVANCE'] > details['PURCHASES']:
        print("Recommendation: Offer short-term loans or debt consolidation plans.")
    elif details['PRC_FULL_PAYMENT'] > 0.8:
        print("Recommendation: Suggest premium wealth management services.")
    elif details['PURCHASES_FREQUENCY'] > 0.5:
        print("Recommendation: Introduce cashback or rewards programs.")
    else:
        print("Recommendation: Focus on improving customer engagement with personalized offers.")


