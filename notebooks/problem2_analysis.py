import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import seaborn as sns

# Load the dataset using the correct relative path
df_q2 = pd.read_csv("../data/Question 2 Datasets _ae16d723e74290c04b72278dc959de6b.csv")

# Part a) Clustering for unlabeled data
unlabeled_data = df_q2[df_q2['Is_Fraud (Labeled Subset)'] == -1.0].copy()
unlabeled_data.reset_index(drop=True, inplace=True)
unlabeled_data = unlabeled_data.drop(['Index', 'Is_Fraud (Labeled Subset)'], axis=1)

categorical_features = ['Location', 'Merchant']
numerical_features = ['Amount', 'Time_Hour']

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = ohe.fit_transform(unlabeled_data[categorical_features])
encoded_df = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out(categorical_features))

scaler = StandardScaler()
scaled_numerical_features = scaler.fit_transform(unlabeled_data[numerical_features])
scaled_numerical_df = pd.DataFrame(scaled_numerical_features, columns=numerical_features)

X_clustering = pd.concat([scaled_numerical_df, encoded_df], axis=1)

# Elbow Method to find optimal k
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_clustering)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.grid(True)
plt.savefig('../output/elbow_method.png')

# Fit K-Means with optimal k (e.g., k=4)
k_optimal = 4
kmeans = KMeans(n_clusters=k_optimal, init='k-means++', random_state=42, n_init=10)
unlabeled_data['Cluster'] = kmeans.fit_predict(X_clustering)

# Visualize clusters with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clustering)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Cluster'] = unlabeled_data['Cluster']

plt.figure(figsize=(10, 8))
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['Cluster'], cmap='viridis', s=50)
plt.title('K-Means Clusters (PCA-Reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(*scatter.legend_elements(), title="Clusters")
plt.grid(True)
plt.savefig('../output/kmeans_clusters.png')

# Part b, c, d) Classification for labeled data
labeled_data = df_q2[df_q2['Is_Fraud (Labeled Subset)'] != -1.0].copy()
labeled_data['Is_Fraud'] = labeled_data['Is_Fraud (Labeled Subset)'].astype(int)
labeled_data = labeled_data.drop(['Index', 'Is_Fraud (Labeled Subset)'], axis=1)

X_labeled = labeled_data.drop('Is_Fraud', axis=1)
y_labeled = labeled_data['Is_Fraud']

# One-Hot Encode and Scale
encoded_features_labeled = ohe.fit_transform(X_labeled[categorical_features])
encoded_df_labeled = pd.DataFrame(encoded_features_labeled, columns=ohe.get_feature_names_out(categorical_features), index=X_labeled.index)
scaled_numerical_features_labeled = scaler.fit_transform(X_labeled[numerical_features])
scaled_numerical_df_labeled = pd.DataFrame(scaled_numerical_features_labeled, columns=numerical_features, index=X_labeled.index)

X_processed = pd.concat([scaled_numerical_df_labeled, encoded_df_labeled], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_processed, y_labeled, test_size=0.3, random_state=42, stratify=y_labeled)

# Initialize Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)

# Calculate metrics and plot
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"F1-score on test data: {f1:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("-" * 30)

cv_f1_scores = cross_val_score(nb_model, X_processed, y_labeled, cv=10, scoring='f1')
print(f"Average F1-score from 10-fold cross-validation: {cv_f1_scores.mean():.2f}")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('../output/confusion_matrix.png')