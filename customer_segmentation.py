import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from utils import generate_customer_data

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def show_customer_segmentation():
    st.title("Customer Segmentation")
    st.write("This model demonstrates customer segmentation using different clustering methods.")

    data = generate_customer_data()

    # Normalize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Clustering method selection
    clustering_method = st.selectbox("Select Clustering Method", ["K-Means", "Hierarchical Clustering"])

    if clustering_method == "K-Means":
        n_clusters = st.slider("Number of Clusters", 2, 10, 4)
        model = KMeans(n_clusters=n_clusters, random_state=42)
    else:
        n_clusters = st.slider("Number of Clusters", 2, 10, 4)
        model = AgglomerativeClustering(n_clusters=n_clusters)

    # Fit the model
    labels = model.fit_predict(data_scaled)
    data['Cluster'] = labels

    # Visualize the clusters
    st.subheader("Customer Segments Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(data['Recency'], data['Monetary'], c=data['Cluster'], s=data['Frequency']*5, alpha=0.6, cmap='viridis')
    ax.set_xlabel('Recency (days since last purchase)')
    ax.set_ylabel('Monetary (total spend)')
    ax.set_title('Customer Segments')
    plt.colorbar(scatter, label='Cluster')
    st.pyplot(fig)

    # Display cluster statistics
    st.subheader("Cluster Statistics")
    cluster_stats = data.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).round(2)
    st.write(cluster_stats)

    # Additional analysis
    st.subheader("Cluster Analysis")
    selected_cluster = st.selectbox("Select a cluster for detailed analysis", sorted(data['Cluster'].unique()))
    cluster_data = data[data['Cluster'] == selected_cluster]

    st.write(f"Cluster {selected_cluster} Statistics:")
    st.write(cluster_data.describe())

    # Visualize the selected cluster
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(cluster_data['Recency'], cluster_data['Monetary'], s=cluster_data['Frequency']*5, alpha=0.6)
    ax.set_xlabel('Recency (days since last purchase)')
    ax.set_ylabel('Monetary (total spend)')
    ax.set_title(f'Cluster {selected_cluster} Details')
    st.pyplot(fig)

    # Recommendations based on cluster
    st.subheader("Cluster-based Recommendations")
    if cluster_data['Recency'].mean() > data['Recency'].mean():
        st.write("This cluster has higher than average recency. Consider a re-engagement campaign.")
    if cluster_data['Frequency'].mean() < data['Frequency'].mean():
        st.write("This cluster has lower than average purchase frequency. Consider loyalty programs or promotions.")
    if cluster_data['Monetary'].mean() > data['Monetary'].mean():
        st.write("This cluster has higher than average total spend. Focus on retention and premium offerings.")

    # Elbow method for K-Means
    if clustering_method == "K-Means":
        st.subheader("Elbow Method for K-Means")
        inertias = []
        k_range = range(1, 11)
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(data_scaled)
            inertias.append(km.inertia_)
        
        fig, ax = plt.subplots()
        ax.plot(k_range, inertias, 'bx-')
        ax.set_xlabel('k')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method for Optimal k')
        st.pyplot(fig)
        st.write("The 'elbow' in the graph suggests the optimal number of clusters.")

    # Dendrogram for Hierarchical Clustering
    if clustering_method == "Hierarchical Clustering":
        st.subheader("Dendrogram for Hierarchical Clustering")
        fig, ax = plt.subplots(figsize=(10, 7))
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
        model = model.fit(data_scaled)
        plot_dendrogram(model, truncate_mode='level', p=3, ax=ax)
        ax.set_title('Hierarchical Clustering Dendrogram')
        st.pyplot(fig)
        st.write("The dendrogram shows the hierarchical relationship between clusters.")