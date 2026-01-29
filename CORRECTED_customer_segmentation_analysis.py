"""
Customer Segmentation in Retail Banking Using Unsupervised Learning
Master's Level Data Science Project - CORRECTED VERSION

Dataset columns (actual):
- customer_id, credit_score, country, gender, age, tenure, balance,
  products_number, credit_card, active_member, estimated_salary, churn

This script implements a comprehensive customer segmentation analysis using:
- K-Means Clustering
- Gaussian Mixture Models (GMM)  
- DBSCAN
- Principal Component Analysis (PCA)
- Multiple validation metrics

Author: Academic Research Assistant
Date: 2026-01-19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("CUSTOMER SEGMENTATION IN RETAIL BANKING - UNSUPERVISED LEARNING")
print("="*80)

# ============================================================================
# STEP 1: DATASET UNDERSTANDING
# ============================================================================
print("\n" + "="*80)
print("STEP 1: DATASET UNDERSTANDING")
print("="*80)

# Load dataset
df = pd.read_csv('Bank Customer Churn Prediction.csv')

print(f"\nDataset Shape: {df.shape[0]} observations × {df.shape[1]} variables")
print(f"\nDataset Dimensions:")
print(f"  - Number of observations: {df.shape[0]:,}")
print(f"  - Number of variables: {df.shape[1]}")

print("\n" + "-"*80)
print("Variable Information:")
print("-"*80)
print(df.dtypes)

print("\n" + "-"*80)
print("First 5 Observations:")
print("-"*80)
print(df.head())

print("\n" + "-"*80)
print("Descriptive Statistics:")
print("-"*80)
print(df.describe())

print("\n" + "-"*80)
print("Missing Values:")
print("-"*80)
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values detected.")

# Academic Dataset Description
print("\n" + "="*80)
print("ACADEMIC DATASET DESCRIPTION")
print("="*80)
dataset_description = f"""
The dataset comprises customer records from a retail banking institution,
containing {df.shape[0]:,} observations across {df.shape[1]} variables. The dataset
includes demographic characteristics (age, gender, country), account-related
features (tenure, balance, number of products), and behavioral indicators
(credit card ownership, active membership status, estimated salary). 
Additionally, a binary churn indicator is present, which will be excluded
from the clustering process to maintain the unsupervised nature of the
analysis, and will only be utilized in post-clustering interpretation.

The variables encompass both categorical (country, gender, credit_card,
active_member) and continuous (credit_score, age, tenure, balance,
products_number, estimated_salary) data types, necessitating appropriate
preprocessing techniques prior to clustering analysis.
"""
print(dataset_description)

# Save dataset description
with open('outputs/01_dataset_description.txt', 'w') as f:
    f.write(dataset_description)

# ============================================================================
# STEP 2: FEATURE SELECTION
# ============================================================================
print("\n" + "="*80)
print("STEP 2: FEATURE SELECTION")
print("="*80)

# Exclude non-informative variables
exclude_vars = ['customer_id', 'churn']
print(f"\nExcluded Variables: {exclude_vars}")
print("\nJustification:")
justification = """
  - customer_id: Unique identifier, not suitable for segmentation
  - churn: Target variable for churn prediction; excluded to maintain
            unsupervised learning framework
"""
print(justification)

# Select features for clustering
clustering_features = [col for col in df.columns if col not in exclude_vars]
print(f"\nSelected Features for Clustering ({len(clustering_features)} variables):")
for i, feat in enumerate(clustering_features, 1):
    print(f"  {i}. {feat}")

print("\n" + "-"*80)
print("Academic Justification for Feature Selection:")
print("-"*80)
academic_justification = """
The selected features represent demographic, financial, and behavioral
dimensions critical for retail banking customer segmentation:

1. DEMOGRAPHIC FEATURES (country, gender, age):
   Enable identification of customer segments based on geographic distribution
   and demographic profiles, facilitating region-specific and age-targeted
   marketing strategies.

2. FINANCIAL FEATURES (credit_score, balance, estimated_salary):
   Capture customers' financial health and wealth indicators, essential for
   risk assessment and product customization.

3. BEHAVIORAL FEATURES (tenure, products_number, credit_card, active_member):
   Reflect customer engagement levels, product adoption patterns, and loyalty
   indicators, which are fundamental for retention and cross-selling strategies.

This feature set provides a comprehensive representation of customer
characteristics while excluding identifiers and the target variable to ensure
a purely unsupervised segmentation approach.
"""
print(academic_justification)

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("STEP 3: DATA PREPROCESSING")
print("="*80)

# Create working dataset
df_clustering = df[clustering_features].copy()

# Handle categorical variables
print("\nEncoding Categorical Variables:")
categorical_vars = df_clustering.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical variables: {categorical_vars}")

# One-hot encoding
df_encoded = pd.get_dummies(df_clustering, columns=categorical_vars, drop_first=True)
print(f"\nDataset shape after encoding: {df_encoded.shape}")
print(f"Features after encoding ({len(df_encoded.columns)} total):")
for i, col in enumerate(df_encoded.columns, 1):
    print(f"  {i}. {col}")

# Check missing values
print("\n" + "-"*80)
print("Missing Values Check:")
print("-"*80)
missing_after_selection = df_encoded.isnull().sum()
if missing_after_selection.sum() > 0:
    print(missing_after_selection[missing_after_selection > 0])
    df_encoded = df_encoded.fillna(df_encoded.median())
    print("Missing values imputed using median strategy.")
else:
    print("No missing values detected in selected features.")

# Feature Scaling
print("\n" + "-"*80)
print("Feature Scaling (Standardization):")
print("-"*80)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)
print(f"Scaled feature matrix shape: {X_scaled.shape}")
print(f"Mean of scaled features: {X_scaled.mean():.6f}")
print(f"Standard deviation of scaled features: {X_scaled.std():.6f}")

print("\n" + "-"*80)
print("Academic Justification for Scaling:")
print("-"*80)
scaling_justification = """
Standardization (z-score normalization) is applied to ensure all features
contribute equally to distance-based clustering algorithms. Without scaling,
features with larger magnitudes (e.g., estimated_salary, balance) would
dominate the distance calculations, leading to biased cluster formations.

StandardScaler transforms each feature to have zero mean and unit variance,
thereby eliminating scale-related biases and improving the convergence
properties of iterative algorithms such as K-Means. This preprocessing step
is essential for ensuring that the clustering results reflect genuine
patterns in customer behavior rather than artifacts of measurement scales.
"""
print(scaling_justification)

# ============================================================================
# STEP 4: DIMENSIONALITY REDUCTION (PCA)
# ============================================================================
print("\n" + "="*80)
print("STEP 4: PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("="*80)

# Apply PCA
pca_full = PCA()
pca_full.fit(X_scaled)

# Explained variance
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\nExplained Variance Ratios (All Components):")
for i, (ev, cv) in enumerate(zip(explained_variance, cumulative_variance), 1):
    print(f"  PC{i}: {ev:.4f} (Cumulative: {cv:.4f})")

# Determine number of components
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1

print(f"\nComponents needed for 90% variance: {n_components_90}")
print(f"Components needed for 95% variance: {n_components_95}")

# Use components explaining 90% variance
n_components = n_components_90
print(f"\nSelected number of components: {n_components}")
print(f"Total variance explained: {cumulative_variance[n_components-1]:.4f}")

# Apply PCA with selected components
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA-transformed data shape: {X_pca.shape}")

# Visualization: Scree plot
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance)+1), explained_variance)
plt.xlabel('Principal Component', fontsize=12)
plt.ylabel('Explained Variance Ratio', fontsize=12)
plt.title('Scree Plot - Explained Variance by Component', fontsize=14, fontweight='bold')
plt.xticks(range(1, len(explained_variance)+1))
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o', linewidth=2, markersize=6)
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Variance', linewidth=2)
plt.axhline(y=0.95, color='g', linestyle='--', label='95% Variance', linewidth=2)
plt.xlabel('Number of Components', fontsize=12)
plt.ylabel('Cumulative Explained Variance', fontsize=12)
plt.title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(range(1, len(cumulative_variance)+1))

plt.tight_layout()
plt.savefig('outputs/02_pca_variance_analysis.png', dpi=300, bbox_inches='tight')
print("\nPCA variance analysis plot saved as 'outputs/02_pca_variance_analysis.png'")
plt.close()

print("\n" + "-"*80)
print("Academic Explanation of PCA Role:")
print("-"*80)
pca_explanation = f"""
Principal Component Analysis (PCA) serves multiple critical functions in
clustering analysis. First, it addresses the curse of dimensionality by
reducing the feature space while retaining the majority of variance in the
data. This reduction enhances computational efficiency and mitigates the
distance concentration problem inherent in high-dimensional spaces, where
distance metrics become less discriminative.

Second, PCA transforms potentially correlated features into orthogonal
principal components, eliminating multicollinearity issues that could
distort clustering results. The transformed features represent uncorrelated
directions of maximum variance, providing a more robust basis for identifying
distinct customer segments.

Third, dimensionality reduction improves cluster interpretability by focusing
on the most informative patterns in the data. By retaining {cumulative_variance[n_components-1]*100:.1f}% of
the total variance with {n_components} components (compared to {X_scaled.shape[1]} original
features), we achieve a parsimonious representation that facilitates both
algorithmic performance and human interpretation of the resulting segments.
"""
print(pca_explanation)

# Save PCA explanation
with open('outputs/02_pca_explanation.txt', 'w') as f:
    f.write(pca_explanation)

# ============================================================================
# STEP 5: CLUSTERING ALGORITHMS
# ============================================================================
print("\n" + "="*80)
print("STEP 5: CLUSTERING ALGORITHMS")
print("="*80)

# ============================================================================
# A) K-MEANS CLUSTERING
# ============================================================================
print("\n" + "-"*80)
print("A) K-MEANS CLUSTERING")
print("-"*80)

# Elbow method for optimal k
inertias = []
silhouette_scores = []
k_range = range(2, 11)

print("\nTesting K-Means with different k values:")
for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X_pca)
    inertias.append(kmeans_temp.inertia_)
    sil_score = silhouette_score(X_pca, kmeans_temp.labels_)
    silhouette_scores.append(sil_score)
    print(f"  k={k}: Inertia={kmeans_temp.inertia_:.2f}, Silhouette={sil_score:.4f}")

# Visualize elbow method
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, marker='o', linewidth=2, markersize=8, color='steelblue')
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(k_range)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, marker='o', linewidth=2, markersize=8, color='green')
plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xticks(k_range)

plt.tight_layout()
plt.savefig('outputs/03_kmeans_optimization.png', dpi=300, bbox_inches='tight')
print("\nK-Means optimization plot saved as 'outputs/03_kmeans_optimization.png'")
plt.close()

# Select optimal k (based on silhouette score)
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal k selected: {optimal_k} (based on maximum Silhouette Score)")

# Final K-Means model
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_pca)

print(f"\nK-Means Cluster Sizes:")
unique, counts = np.unique(kmeans_labels, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count:,} customers ({count/len(kmeans_labels)*100:.2f}%)")

# ============================================================================
# B) GAUSSIAN MIXTURE MODELS (GMM)
# ============================================================================
print("\n" + "-"*80)
print("B) GAUSSIAN MIXTURE MODELS (GMM)")
print("-"*80)

# Use same number of clusters as K-Means
n_clusters_gmm = optimal_k
print(f"\nNumber of components for GMM: {n_clusters_gmm}")

gmm = GaussianMixture(n_components=n_clusters_gmm, random_state=42, n_init=10)
gmm_labels = gmm.fit_predict(X_pca)

print(f"\nGMM Cluster Sizes:")
unique, counts = np.unique(gmm_labels, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"  Cluster {cluster}: {count:,} customers ({count/len(gmm_labels)*100:.2f}%)")

print("\n" + "-"*80)
print("GMM Probabilistic Membership:")
print("-"*80)
gmm_explanation = """
Unlike K-Means, which assigns each observation to a single cluster (hard
assignment), Gaussian Mixture Models employ soft assignment by computing
probabilistic membership for each customer across all clusters. This
probabilistic framework is particularly valuable in retail banking, where
customers may exhibit characteristics spanning multiple segments.

The GMM approach models each cluster as a Gaussian distribution, allowing
for more flexible cluster shapes compared to K-Means' assumption of spherical
clusters. This flexibility can better capture the heterogeneity in customer
behavior patterns.
"""
print(gmm_explanation)

# ============================================================================
# C) DBSCAN
# ============================================================================
print("\n" + "-"*80)
print("C) DBSCAN (Density-Based Spatial Clustering)")
print("-"*80)

# Parameter tuning for DBSCAN
print("\nTesting DBSCAN with different parameter combinations:")
eps_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
min_samples_values = [5, 10, 15, 20]

best_dbscan = None
best_score = -1
best_params = {}
dbscan_results = []

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan_temp = DBSCAN(eps=eps, min_samples=min_samples)
        labels_temp = dbscan_temp.fit_predict(X_pca)
        
        n_clusters = len(set(labels_temp)) - (1 if -1 in labels_temp else 0)
        n_noise = list(labels_temp).count(-1)
        
        if n_clusters > 1:
            try:
                mask = labels_temp != -1
                if mask.sum() > n_clusters:
                    score = silhouette_score(X_pca[mask], labels_temp[mask])
                    dbscan_results.append({
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'n_noise': n_noise,
                        'silhouette': score
                    })
                    print(f"  eps={eps}, min_samples={min_samples}: "
                          f"Clusters={n_clusters}, Noise={n_noise}, Silhouette={score:.4f}")
                    
                    if score > best_score and n_noise < len(labels_temp) * 0.3:
                        best_score = score
                        best_dbscan = dbscan_temp
                        best_params = {'eps': eps, 'min_samples': min_samples}
            except:
                pass

if best_dbscan is not None:
    print(f"\nBest DBSCAN parameters: eps={best_params['eps']}, "
          f"min_samples={best_params['min_samples']}")
    dbscan_labels = best_dbscan.labels_
else:
    print("\nUsing default DBSCAN parameters: eps=2.0, min_samples=10")
    dbscan = DBSCAN(eps=2.0, min_samples=10)
    dbscan_labels = dbscan.fit_predict(X_pca)

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"\nDBSCAN Results:")
print(f"  Number of clusters: {n_clusters_dbscan}")
print(f"  Number of noise points (outliers): {n_noise} ({n_noise/len(dbscan_labels)*100:.2f}%)")

print(f"\nDBSCAN Cluster Sizes:")
unique, counts = np.unique(dbscan_labels, return_counts=True)
for cluster, count in zip(unique, counts):
    if cluster == -1:
        print(f"  Noise/Outliers: {count:,} customers ({count/len(dbscan_labels)*100:.2f}%)")
    else:
        print(f"  Cluster {cluster}: {count:,} customers ({count/len(dbscan_labels)*100:.2f}%)")

print("\n" + "-"*80)
print("DBSCAN Outlier Interpretation in Retail Banking:")
print("-"*80)
dbscan_interpretation = """
In the context of retail banking, customers identified as outliers (noise
points) by DBSCAN represent individuals with atypical behavioral patterns
that do not conform to any major customer segment. These outliers may include:

1. High-net-worth individuals with unique financial profiles
2. Customers with unusual product combinations or usage patterns
3. Recently acquired customers still establishing their banking relationships
4. Customers in transitional life stages (e.g., recent graduates, retirees)

From a business perspective, outliers warrant special attention as they may
represent either high-value opportunities requiring personalized service or
potential fraud/anomaly cases requiring investigation. The identification of
such customers is a unique advantage of density-based clustering approaches
like DBSCAN.
"""
print(dbscan_interpretation)

# ============================================================================
# STEP 6: CLUSTER VALIDATION & COMPARISON
# ============================================================================
print("\n" + "="*80)
print("STEP 6: CLUSTER VALIDATION & COMPARISON")
print("="*80)

# Calculate validation metrics
validation_results = []

# K-Means metrics
kmeans_sil = silhouette_score(X_pca, kmeans_labels)
kmeans_ch = calinski_harabasz_score(X_pca, kmeans_labels)
kmeans_db = davies_bouldin_score(X_pca, kmeans_labels)

validation_results.append({
    'Algorithm': 'K-Means',
    'Silhouette Score': kmeans_sil,
    'Calinski-Harabasz Index': kmeans_ch,
    'Davies-Bouldin Index': kmeans_db,
    'Number of Clusters': optimal_k
})

# GMM metrics
gmm_sil = silhouette_score(X_pca, gmm_labels)
gmm_ch = calinski_harabasz_score(X_pca, gmm_labels)
gmm_db = davies_bouldin_score(X_pca, gmm_labels)

validation_results.append({
    'Algorithm': 'GMM',
    'Silhouette Score': gmm_sil,
    'Calinski-Harabasz Index': gmm_ch,
    'Davies-Bouldin Index': gmm_db,
    'Number of Clusters': n_clusters_gmm
})

# DBSCAN metrics (excluding noise)
if n_clusters_dbscan > 1:
    mask_dbscan = dbscan_labels != -1
    if mask_dbscan.sum() > n_clusters_dbscan:
        dbscan_sil = silhouette_score(X_pca[mask_dbscan], dbscan_labels[mask_dbscan])
        dbscan_ch = calinski_harabasz_score(X_pca[mask_dbscan], dbscan_labels[mask_dbscan])
        dbscan_db = davies_bouldin_score(X_pca[mask_dbscan], dbscan_labels[mask_dbscan])
        
        validation_results.append({
            'Algorithm': 'DBSCAN',
            'Silhouette Score': dbscan_sil,
            'Calinski-Harabasz Index': dbscan_ch,
            'Davies-Bouldin Index': dbscan_db,
            'Number of Clusters': n_clusters_dbscan
        })

# Create comparison table
validation_df = pd.DataFrame(validation_results)

print("\n" + "-"*80)
print("COMPARATIVE VALIDATION METRICS TABLE")
print("-"*80)
print(validation_df.to_string(index=False))

# Save to CSV
validation_df.to_csv('outputs/04_cluster_validation_metrics.csv', index=False)
print("\nValidation metrics saved to 'outputs/04_cluster_validation_metrics.csv'")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

metrics = ['Silhouette Score', 'Calinski-Harabasz Index', 'Davies-Bouldin Index']
colors = ['steelblue', 'green', 'coral']

for idx, (metric, color) in enumerate(zip(metrics, colors)):
    ax = axes[idx]
    values = validation_df[metric].values
    algorithms = validation_df['Algorithm'].values
    
    bars = ax.bar(algorithms, values, color=color, alpha=0.7, edgecolor='black')
    ax.set_ylabel(metric, fontsize=11, fontweight='bold')
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/04_validation_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("Validation metrics visualization saved as 'outputs/04_validation_metrics_comparison.png'")
plt.close()

print("\n" + "-"*80)
print("INTERPRETATION OF VALIDATION METRICS:")
print("-"*80)
metrics_interpretation = """
1. SILHOUETTE SCORE (Range: -1 to +1, Higher is Better):
   Measures how similar an object is to its own cluster compared to other
   clusters. Values closer to +1 indicate well-separated, compact clusters.

2. CALINSKI-HARABASZ INDEX (Higher is Better):
   Ratio of between-cluster dispersion to within-cluster dispersion. Higher
   values indicate better-defined clusters with greater separation.

3. DAVIES-BOULDIN INDEX (Lower is Better):
   Average similarity ratio of each cluster with its most similar cluster.
   Lower values indicate better cluster separation.
"""
print(metrics_interpretation)

# Determine best algorithm
print("\n" + "-"*80)
print("BEST PERFORMING ALGORITHM ANALYSIS:")
print("-"*80)

# Normalize metrics
validation_df_norm = validation_df.copy()
validation_df_norm['Sil_Norm'] = validation_df_norm['Silhouette Score']
validation_df_norm['CH_Norm'] = (
    validation_df_norm['Calinski-Harabasz Index'] / validation_df_norm['Calinski-Harabasz Index'].max()
)
validation_df_norm['DB_Norm'] = (
    1 - (validation_df_norm['Davies-Bouldin Index'] / validation_df_norm['Davies-Bouldin Index'].max())
)

validation_df_norm['Overall Score'] = (
    validation_df_norm['Sil_Norm'] +
    validation_df_norm['CH_Norm'] +
    validation_df_norm['DB_Norm']
) / 3

print(validation_df_norm[['Algorithm', 'Overall Score']].to_string(index=False))

best_algorithm = validation_df_norm.loc[validation_df_norm['Overall Score'].idxmax(), 'Algorithm']
print(f"\nBest Overall Algorithm: {best_algorithm}")

# ============================================================================
# STEP 7: CLUSTER PROFILING
# ============================================================================
print("\n" + "="*80)
print("STEP 7: CLUSTER PROFILING")
print("="*80)

# Use K-Means for detailed profiling
df_profiling = df.copy()
df_profiling['Cluster_KMeans'] = kmeans_labels
df_profiling['Cluster_GMM'] = gmm_labels
df_profiling['Cluster_DBSCAN'] = dbscan_labels

print("\nDETAILED CLUSTER PROFILING (K-MEANS)")
print("-"*80)

# Profile features
profile_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary']

cluster_profiles = []

for cluster_id in range(optimal_k):
    cluster_data = df_profiling[df_profiling['Cluster_KMeans'] == cluster_id]
    
    print(f"\n{'='*80}")
    print(f"CLUSTER {cluster_id} (n={len(cluster_data):,}, {len(cluster_data)/len(df)*100:.2f}%)")
    print(f"{'='*80}")
    
    # Continuous variables
    print("\nContinuous Variables (Mean ± Std):")
    for var in profile_features:
        mean_val = cluster_data[var].mean()
        std_val = cluster_data[var].std()
        median_val = cluster_data[var].median()
        print(f"  {var:20s}: Mean={mean_val:12,.2f}, Median={median_val:12,.2f}, Std={std_val:10,.2f}")
    
    # Categorical variables
    print("\nCategorical Variables (Proportions):")
    print(f"  Country:")
    country_dist = cluster_data['country'].value_counts(normalize=True)
    for country, prop in country_dist.items():
        print(f"    {country}: {prop:.2%}")
    
    print(f"  Gender:")
    gender_dist = cluster_data['gender'].value_counts(normalize=True)
    for gender, prop in gender_dist.items():
        print(f"    {gender}: {prop:.2%}")
    
    print(f"  Has Credit Card: {cluster_data['credit_card'].mean():.2%}")
    print(f"  Is Active Member: {cluster_data['active_member'].mean():.2%}")
    
    # Store profile
    profile = {
        'Cluster': cluster_id,
        'Size': len(cluster_data),
        'Percentage': f"{len(cluster_data)/len(df)*100:.2f}%",
        'Avg_Age': cluster_data['age'].mean(),
        'Avg_CreditScore': cluster_data['credit_score'].mean(),
        'Avg_Balance': cluster_data['balance'].mean(),
        'Avg_Tenure': cluster_data['tenure'].mean(),
        'Avg_NumProducts': cluster_data['products_number'].mean(),
        'Avg_Salary': cluster_data['estimated_salary'].mean(),
        'Active_Member_%': cluster_data['active_member'].mean() * 100,
        'Has_CreditCard_%': cluster_data['credit_card'].mean() * 100,
        'Female_%': (cluster_data['gender'] == 'Female').mean() * 100
    }
    cluster_profiles.append(profile)

profile_df = pd.DataFrame(cluster_profiles)
print("\n" + "="*80)
print("CLUSTER PROFILING SUMMARY TABLE")
print("="*80)
print(profile_df.to_string(index=False))

profile_df.to_csv('outputs/05_cluster_profiles_kmeans.csv', index=False)
print("\nCluster profiles saved to 'outputs/05_cluster_profiles_kmeans.csv'")

# Business segment naming
print("\n" + "="*80)
print("BUSINESS-ORIENTED SEGMENT NAMING & INTERPRETATION")
print("="*80)

segment_names = {}
segment_descriptions = {}

for cluster_id in range(optimal_k):
    cluster_data = df_profiling[df_profiling['Cluster_KMeans'] == cluster_id]
    
    avg_age = cluster_data['age'].mean()
    avg_balance = cluster_data['balance'].mean()
    avg_products = cluster_data['products_number'].mean()
    avg_tenure = cluster_data['tenure'].mean()
    active_rate = cluster_data['active_member'].mean()
    avg_salary = cluster_data['estimated_salary'].mean()
    avg_credit_score = cluster_data['credit_score'].mean()
    
    # Segment naming logic
    if avg_balance > df['balance'].quantile(0.75) and avg_products >= 2:
        name = "Premium Engaged Customers"
        desc = """High-value customers with substantial account balances and
        multiple product holdings. These customers represent the bank's most
        profitable segment and should receive premium service offerings."""
    elif avg_age < 35 and avg_tenure < 3:
        name = "Young Emerging Customers"
        desc = """Younger customers in early stages of their banking relationship.
        This segment represents future growth potential and should be targeted
        with digital-first products and financial education programs."""
    elif active_rate < 0.3 and avg_tenure > 5:
        name = "Dormant Long-Tenure Customers"
        desc = """Long-standing customers with low engagement levels. This segment
        requires reactivation campaigns and personalized outreach to prevent
        attrition and restore relationship value."""
    elif avg_products < 1.5 and avg_balance < df['balance'].median():
        name = "Basic Service Customers"
        desc = """Customers with minimal product adoption and modest balances.
        This segment represents cross-selling opportunities for additional
        products and services."""
    elif avg_age > 45 and avg_balance > df['balance'].median():
        name = "Mature Stable Customers"
        desc = """Middle-aged to senior customers with stable financial profiles.
        This segment values reliability and should be offered wealth management
        and retirement planning services."""
    else:
        name = f"Standard Segment {cluster_id}"
        desc = """Customers with average characteristics across multiple dimensions.
        This segment requires standard service levels with opportunities for
        targeted product recommendations."""
    
    segment_names[cluster_id] = name
    segment_descriptions[cluster_id] = desc
    
    print(f"\nCLUSTER {cluster_id}: {name}")
    print("-" * 80)
    print(desc)
    print(f"\nKey Characteristics:")
    print(f"  - Average Age: {avg_age:.1f} years")
    print(f"  - Average Credit Score: {avg_credit_score:.0f}")
    print(f"  - Average Balance: ${avg_balance:,.2f}")
    print(f"  - Average Products: {avg_products:.2f}")
    print(f"  - Average Tenure: {avg_tenure:.1f} years")
    print(f"  - Active Member Rate: {active_rate:.2%}")
    print(f"  - Average Salary: ${avg_salary:,.2f}")

# Save segment descriptions
with open('outputs/05_segment_descriptions.txt', 'w') as f:
    for cluster_id in range(optimal_k):
        f.write(f"CLUSTER {cluster_id}: {segment_names[cluster_id]}\n")
        f.write("="*80 + "\n")
        f.write(segment_descriptions[cluster_id] + "\n\n")

# ============================================================================
# STEP 8: POST-CLUSTERING CHURN ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("STEP 8: POST-CLUSTERING CHURN ANALYSIS")
print("="*80)

print("\nReintroducing 'churn' variable for post-clustering analysis...")
print("Note: This variable was NOT used during clustering to maintain")
print("unsupervised learning framework.\n")

# Churn analysis by cluster
churn_analysis = []

for cluster_id in range(optimal_k):
    cluster_data = df_profiling[df_profiling['Cluster_KMeans'] == cluster_id]
    churn_rate = cluster_data['churn'].mean()
    n_churned = cluster_data['churn'].sum()
    n_total = len(cluster_data)
    
    churn_analysis.append({
        'Cluster': cluster_id,
        'Segment_Name': segment_names[cluster_id],
        'Total_Customers': n_total,
        'Churned_Customers': n_churned,
        'Churn_Rate': churn_rate,
        'Churn_Rate_%': f"{churn_rate*100:.2f}%"
    })
    
    print(f"Cluster {cluster_id} ({segment_names[cluster_id]}):")
    print(f"  Total Customers: {n_total:,}")
    print(f"  Churned Customers: {n_churned:,}")
    print(f"  Churn Rate: {churn_rate:.2%}")
    print()

churn_df = pd.DataFrame(churn_analysis)
print("-" * 80)
print("CHURN RATE BY CLUSTER SUMMARY")
print("-" * 80)
print(churn_df[['Cluster', 'Segment_Name', 'Total_Customers', 'Churn_Rate_%']].to_string(index=False))

churn_df.to_csv('outputs/06_churn_analysis_by_cluster.csv', index=False)
print("\nChurn analysis saved to 'outputs/06_churn_analysis_by_cluster.csv'")

# Statistical significance
print("\n" + "-" * 80)
print("STATISTICAL ANALYSIS OF CHURN DIFFERENCES")
print("-" * 80)

contingency_table = pd.crosstab(df_profiling['Cluster_KMeans'], df_profiling['churn'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print(f"\nChi-Square Test for Independence:")
print(f"  Chi-Square Statistic: {chi2:.4f}")
print(f"  P-value: {p_value:.6f}")
print(f"  Degrees of Freedom: {dof}")

if p_value < 0.05:
    print("\n  Result: Churn rates differ significantly across clusters (p < 0.05)")
else:
    print("\n  Result: No significant difference in churn rates across clusters (p >= 0.05)")

# Visualization
plt.figure(figsize=(12, 6))
clusters_sorted = churn_df.sort_values('Churn_Rate', ascending=False)

bars = plt.barh(range(len(clusters_sorted)), clusters_sorted['Churn_Rate'] * 100, 
                color='coral', edgecolor='black', alpha=0.7)

plt.yticks(range(len(clusters_sorted)), 
           [f"Cluster {c}\n{n[:30]}..." if len(n) > 30 else f"Cluster {c}\n{n}" 
            for c, n in zip(clusters_sorted['Cluster'], clusters_sorted['Segment_Name'])],
           fontsize=10)
plt.xlabel('Churn Rate (%)', fontsize=12, fontweight='bold')
plt.title('Churn Rate by Customer Segment', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2., 
             f'{width:.2f}%',
             ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/06_churn_rate_by_cluster.png', dpi=300, bbox_inches='tight')
print("\nChurn rate visualization saved as 'outputs/06_churn_rate_by_cluster.png'")
plt.close()

# ============================================================================
# STEP 9: ADDITIONAL HIGH-LEVEL ANALYSES
# ============================================================================
print("\n" + "="*80)
print("STEP 9: ADDITIONAL HIGH-LEVEL ANALYSES")
print("="*80)

# A) Stability comparison
print("\n" + "-" * 80)
print("A) STABILITY COMPARISON: K-MEANS vs GMM")
print("-" * 80)

n_iterations = 10
kmeans_stability = []
gmm_stability = []

print(f"\nRunning {n_iterations} iterations with different random seeds...")

for seed in range(n_iterations):
    km_temp = KMeans(n_clusters=optimal_k, random_state=seed, n_init=10)
    km_labels_temp = km_temp.fit_predict(X_pca)
    kmeans_stability.append(km_labels_temp)
    
    gmm_temp = GaussianMixture(n_components=optimal_k, random_state=seed, n_init=10)
    gmm_labels_temp = gmm_temp.fit_predict(X_pca)
    gmm_stability.append(gmm_labels_temp)

# Calculate ARI
kmeans_ari_scores = []
gmm_ari_scores = []

print("\nAdjusted Rand Index (ARI) between consecutive iterations:")
for i in range(n_iterations - 1):
    kmeans_ari = adjusted_rand_score(kmeans_stability[i], kmeans_stability[i+1])
    gmm_ari = adjusted_rand_score(gmm_stability[i], gmm_stability[i+1])
    
    kmeans_ari_scores.append(kmeans_ari)
    gmm_ari_scores.append(gmm_ari)
    
    print(f"  Iteration {i} vs {i+1}: K-Means ARI={kmeans_ari:.4f}, GMM ARI={gmm_ari:.4f}")

print(f"\nAverage Stability (ARI):")
print(f"  K-Means: {np.mean(kmeans_ari_scores):.4f} ± {np.std(kmeans_ari_scores):.4f}")
print(f"  GMM:     {np.mean(gmm_ari_scores):.4f} ± {np.std(gmm_ari_scores):.4f}")

if np.mean(kmeans_ari_scores) > np.mean(gmm_ari_scores):
    print("\n  Result: K-Means demonstrates higher stability across random initializations")
else:
    print("\n  Result: GMM demonstrates higher stability across random initializations")

# Save stability results
stability_df = pd.DataFrame({
    'Iteration': range(n_iterations-1),
    'KMeans_ARI': kmeans_ari_scores,
    'GMM_ARI': gmm_ari_scores
})
stability_df.to_csv('outputs/07_stability_analysis.csv', index=False)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated Outputs:")
print("  1. 01_dataset_description.txt")
print("  2. 02_pca_variance_analysis.png")
print("  3. 02_pca_explanation.txt")
print("  4. 03_kmeans_optimization.png")
print("  5. 04_cluster_validation_metrics.csv")
print("  6. 04_validation_metrics_comparison.png")
print("  7. 05_cluster_profiles_kmeans.csv")
print("  8. 05_segment_descriptions.txt")
print("  9. 06_churn_analysis_by_cluster.csv")
print(" 10. 06_churn_rate_by_cluster.png")
print(" 11. 07_stability_analysis.csv")
print("\nNext: Academic output generation (Step 10)")
