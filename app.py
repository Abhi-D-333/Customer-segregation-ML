
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("📊 Customer Segmentation Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload customer CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.write(df.head())

    features = ['Age', 'Income', 'SpendingScore', 'PurchaseFrequency']

    if all(col in df.columns for col in features):
        n_clusters = st.slider("Select number of customer segments", 2, 8, 4)

        # Preprocessing
        X = df[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        df['Cluster'] = labels

        # PCA for visualization
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        df['PC1'] = components[:, 0]
        df['PC2'] = components[:, 1]

        st.subheader("Clustered Data")
        st.write(df[['Age', 'Income', 'SpendingScore', 'PurchaseFrequency', 'Cluster']].head())

        # Plot clusters
        st.subheader("🌀 Cluster Visualization (PCA Projection)")
        fig, ax = plt.subplots()
        scatter = ax.scatter(df['PC1'], df['PC2'], c=df['Cluster'], cmap='tab10', s=60)
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_title('Customer Segments')
        st.pyplot(fig)

        # Cluster stats
        st.subheader("📈 Segment Statistics")
        cluster_summary = df.groupby('Cluster')[features].mean().round(2)
        st.dataframe(cluster_summary)

    else:
        st.error(f"CSV must contain the columns: {features}")
else:
    st.info("Please upload a CSV file with columns: Age, Income, SpendingScore, PurchaseFrequency")
