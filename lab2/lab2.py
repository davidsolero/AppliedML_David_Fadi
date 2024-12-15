import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
# **1. Carga de Datos**
# Cargar el dataset
data = pd.read_csv('Customer Data Fadi.csv')

# Eliminar el atributo de identificación
if "CUST_ID" in data.columns:
    data = data.drop(columns=["CUST_ID"])

# **2. Preprocesamiento de Datos**
# Manejar valores faltantes
imputer = SimpleImputer(strategy='median')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Estandarizar los datos
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data_imputed), columns=data.columns)

# **3. Clustering Jerárquico**
# Crear el dendrograma
linkage_matrix = linkage(data_scaled, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()

# Seleccionar el número óptimo de clústeres cortando el dendrograma
num_clusters = 5  # Define un número razonable de clústeres
clusters_hierarchical = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')

# Agregar etiquetas de clúster a los datos
data['Cluster_Hierarchical'] = clusters_hierarchical

# **4. DBSCAN**
# Aplicar DBSCAN
dbscan = DBSCAN(eps=2, min_samples=20)
clusters_dbscan = dbscan.fit_predict(data_scaled)

# Agregar etiquetas de clúster a los datos
data['Cluster_DBSCAN'] = clusters_dbscan

# Evaluar el rendimiento con Silhouette Score
if len(set(clusters_dbscan)) > 1:  # Evitar escenarios de un solo clúster
    sil_score = silhouette_score(data_scaled, clusters_dbscan)
    print(f"Silhouette Score for DBSCAN: {sil_score}")
else:
    print("DBSCAN produced a single cluster.")

# **5. Visualización**
# Visualización con PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(10, 7))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters_dbscan, cmap='viridis', s=10)
plt.title("DBSCAN Clusters Visualized with PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()

# Visualización con t-SNE
tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(data_scaled)

plt.figure(figsize=(10, 7))
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=clusters_dbscan, cmap='viridis', s=10)
plt.title("DBSCAN Clusters Visualized with t-SNE")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.colorbar(label="Cluster")
plt.show()

# **6. Análisis de Clústeres y Recomendaciones**
cluster_summary = data.groupby('Cluster_DBSCAN').mean()

print("Cluster Recommendations:")
for cluster, details in cluster_summary.iterrows():
    print(f"\nCluster {cluster}:")

    # Recomendaciones basadas en los valores promedio del clúster
    if details['CASH_ADVANCE'] > details['PURCHASES']:
        print("Recommendation: Offer short-term loans or debt consolidation plans.")
    elif details['PRC_FULL_PAYMENT'] > 0.8:
        print("Recommendation: Suggest premium wealth management services.")
    elif details['PURCHASES_FREQUENCY'] > 0.5:
        print("Recommendation: Introduce cashback or rewards programs.")
    elif details['INSTALLMENTS_PURCHASES'] > details['ONEOFF_PURCHASES']:
        print("Recommendation: Promote installment payment options or EMIs.")
    else:
        print("Recommendation: Focus on personalized offers to increase engagement.")
