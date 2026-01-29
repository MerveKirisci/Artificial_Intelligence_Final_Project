# Customer Segmentation in Retail Banking Using Unsupervised Learning

A comprehensive Master's level data science project implementing customer segmentation analysis using K-Means, Gaussian Mixture Models (GMM), and DBSCAN clustering algorithms.

## 📊 Project Overview

This project analyzes 10,000 retail banking customers to identify distinct customer segments using unsupervised machine learning techniques. The analysis provides actionable insights for customer relationship management, retention strategies, and cross-selling opportunities.

### Key Findings

- **3 Distinct Customer Segments** identified with varying churn risks (16.07% - 25.03%)
- **Product diversification** reduces churn by 56% compared to single-product customers
- **High-balance customers** paradoxically show highest churn (25.03%), indicating balance alone doesn't ensure loyalty
- **Statistical significance** confirmed via chi-square test (χ²(2) = 115.84, p < 0.001)

## 🎯 Business Impact

### Identified Segments

1. **Multi-Product Engaged Customers** (28.93%)
   - Highest product adoption (1.81 products)
   - Lowest churn rate (16.07%)
   - Strong loyalty through diversification

2. **Moderate Balance Customers** (24.77%)
   - Balanced financial profile
   - Growth potential in products and balances
   - Moderate churn (16.67%)

3. **High-Balance At-Risk Customers** (46.30%)
   - Highest balances ($123,132 avg)
   - Lowest product adoption (1.35 products)
   - **Critical churn risk** (25.03%)

## � Quick Start

### Prerequisites

- **Python 3.9+** (tested on Python 3.9-3.11)
- Required packages listed in `requirements.txt`

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <repo-directory>
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis

**Single command execution:**

```bash
python3 CORRECTED_customer_segmentation_analysis.py
```

**Expected Runtime:** 2-5 minutes

**Outputs:** 11 files generated in `outputs/` directory:
- 4 visualizations (PNG)
- 4 data tables (CSV)
- 3 text descriptions (TXT)

## 📁 Repository Structure

```
.
├── CORRECTED_customer_segmentation_analysis.py  # Main analysis script
├── Bank Customer Churn Prediction.csv           # Dataset (10,000 customers)
├── ACADEMIC_OUTPUT_Customer_Segmentation.md     # Complete research report
├── requirements.txt                              # Python dependencies
├── .gitignore                                    # Git ignore rules
├── README.md                                     # This file
│
└── outputs/                                      # Generated outputs
    ├── 01_dataset_description.txt
    ├── 02_pca_variance_analysis.png
    ├── 02_pca_explanation.txt
    ├── 03_kmeans_optimization.png
    ├── 04_cluster_validation_metrics.csv
    ├── 04_validation_metrics_comparison.png
    ├── 05_cluster_profiles_kmeans.csv
    ├── 05_segment_descriptions.txt
    ├── 06_churn_analysis_by_cluster.csv
    ├── 06_churn_rate_by_cluster.png
    └── 07_stability_analysis.csv
```

## 🔬 Methodology

### Algorithms Implemented

- **K-Means Clustering** (selected as optimal)
  - Silhouette Score: 0.1252
  - Calinski-Harabasz Index: 1206.73
  - Davies-Bouldin Index: 2.417
  - Stability (ARI): 0.786 ± 0.238

- **Gaussian Mixture Models (GMM)**
  - Probabilistic cluster assignments
  - Stability (ARI): 0.540 ± 0.344

- **DBSCAN**
  - Density-based clustering
  - Outlier detection capability

### Validation Metrics

- Silhouette Score (cluster cohesion and separation)
- Calinski-Harabasz Index (between/within cluster dispersion)
- Davies-Bouldin Index (cluster similarity)
- Adjusted Rand Index (stability analysis)

### Dimensionality Reduction

- Principal Component Analysis (PCA)
- 90% variance retained with reduced dimensions
- Enhanced clustering robustness

## 📈 Results Summary

### Validation Metrics Comparison

| Algorithm | Silhouette | Calinski-Harabasz | Davies-Bouldin | Clusters |
|-----------|------------|-------------------|----------------|----------|
| K-Means   | 0.1252     | 1206.73          | 2.417          | 3        |
| GMM       | 0.1219     | 1187.37          | 2.534          | 3        |
| DBSCAN    | 0.1344     | 432.89           | 2.353          | 16       |

**Winner:** K-Means (optimal balance of performance, stability, and interpretability)

### Churn Analysis

| Segment | Size | Churn Rate | Risk Level |
|---------|------|------------|------------|
| Multi-Product Engaged | 2,893 (28.93%) | 16.07% | Low |
| Moderate Balance | 2,477 (24.77%) | 16.67% | Moderate |
| High-Balance At-Risk | 4,630 (46.30%) | 25.03% | **High** |

## 💼 Strategic Recommendations

### For High-Balance At-Risk Customers (Critical Priority)
- Implement aggressive retention programs
- Assign dedicated relationship managers
- Offer premium products to increase stickiness
- Provide incentives for product adoption

### For Multi-Product Engaged Customers
- Leverage as model for cross-selling campaigns
- Implement loyalty rewards tied to product holdings
- Focus on relationship deepening

### For Moderate Balance Customers
- Targeted cross-selling to increase products from 1.54 to 2+
- Offer investment products and savings accounts
- Financial planning services to deepen relationships

## 📊 Visualizations

The analysis generates 4 key visualizations in `outputs/`:

1. **PCA Variance Analysis** - Scree plot and cumulative variance
2. **K-Means Optimization** - Elbow method and silhouette scores
3. **Validation Metrics Comparison** - Algorithm performance
4. **Churn Rate by Segment** - Risk stratification

## 🎓 Academic Rigor

- **Unsupervised Learning Framework:** Churn variable excluded from clustering
- **Multi-Algorithm Comparison:** 3 algorithms with 3 validation metrics each
- **Statistical Testing:** Chi-square test confirms significance (p < 0.001)
- **Stability Analysis:** K-Means demonstrates 45% higher stability than GMM
- **Reproducibility:** Fixed random seeds (random_state=42)

## 📄 Documentation

- **ACADEMIC_OUTPUT_Customer_Segmentation.md** - Complete research report (10,000+ words)
  - Methodology section with mathematical formulations
  - Results and evaluation with all metrics
  - Discussion and managerial implications
  - Conclusion and future research directions

## 🔑 Key Insights

1. **Product Diversification Drives Retention**
   - Customers with multiple products have 56% lower churn
   - Product adoption is more predictive of loyalty than balance

2. **Balance ≠ Loyalty**
   - Highest-balance segment has highest churn rate
   - Relationship depth matters more than account size

3. **Actionable Segmentation**
   - 3 distinct segments enable targeted strategies
   - Churn variation of 1.56x across segments enables risk-based allocation

4. **Algorithm Selection**
   - K-Means optimal for operational deployment
   - Superior stability (ARI=0.786) ensures consistent segmentation

## 📚 Dataset

**Source:** Bank Customer Churn Prediction dataset  
**Size:** 10,000 customers × 12 variables  
**Included:** Yes (`Bank Customer Churn Prediction.csv`)

**Variables:**
- Demographics: age, gender, country
- Financial: credit_score, balance, estimated_salary
- Behavioral: tenure, products_number, credit_card, active_member
- Target: churn (excluded from clustering, used only in post-hoc analysis)

## 🔧 Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError`  
**Solution:** Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue:** `FileNotFoundError` for dataset  
**Solution:** Ensure `Bank Customer Churn Prediction.csv` is in the same directory as the script

**Issue:** Permission denied for `outputs/` directory  
**Solution:** The script creates the directory automatically. Ensure write permissions in the project folder.

## 📚 References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
- Kaufman, L., & Rousseeuw, P. J. (2005). *Finding Groups in Data*. John Wiley & Sons.
- McLachlan, G. J., & Peel, D. (2000). *Finite Mixture Models*. John Wiley & Sons.
- Ester, M., et al. (1996). A density-based algorithm for discovering clusters. *KDD*.

## 📝 License

This project is for academic and educational purposes.

## 👤 Author

Master's Level Data Science Project - Customer Segmentation in Retail Banking

---

**Project Status:** ✅ Complete  
**Analysis Date:** January 2026  
**Tested on:** Python 3.9, 3.10, 3.11  
**Reproducibility:** Fully reproducible with fixed random seeds
