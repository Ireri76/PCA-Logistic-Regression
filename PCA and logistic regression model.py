#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Step 1: LIBRARIES LOADING-the required libraries
import pandas as pd # Data manipulation, analysis
import plotly.express as px # Data visualization, plotting

from sklearn.datasets import load_breast_cancer # Data loading utility
from sklearn.cluster import KMeans # Clustering algorithm
from sklearn.decomposition import PCA # Dimensionality reduction
from sklearn.metrics import silhouette_score # Cluster evaluation
from sklearn.pipeline import make_pipeline # Model pipeline creation
from sklearn.preprocessing import StandardScaler # Feature scaling normalization
from sklearn.utils.validation import check_is_fitted # Model validation check
from sklearn.linear_model import LogisticRegression # Classification model
from sklearn.model_selection import train_test_split # Data splitting
from sklearn.metrics import classification_report, confusion_matrix # Model performance report & Model evaluation matrix
from scipy.stats.mstats import trimmed_var # Trimmed variance calculation


# In[2]:


# Step 2: DATA LOADING-the breast cancer dataset
cancer_data = load_breast_cancer()

# Create a DataFrame from the cancer dataset
df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)

# Add the target labels to the DataFrame
df['Target'] = cancer_data.target

# Display basic information about the dataset
print("df type:", type(df))
print("df shape:", df.shape)
df.head()


# In[3]:


# Step 3: DATA EXPLORATION-Calculate variances for all features and identify the top 10 features with the highest variances
var = df.var().sort_values().tail(10)
print("Top 10 variances type:", type(var))
print("Top 10 variances shape:", var.shape)
print(var)


# In[4]:


# Step 4: DATA VISUALIZATION-Visualize the top 10 variances using a horizontal bar chart
fig = px.bar(
    x=var,
    y=var.index,
    title="Cancer: High Variance Features"
)
fig.update_layout(xaxis_title="Variance", yaxis_title="Feature", xaxis_tickformat='.2f')
fig.show()

# Create boxplots for specific features which had the highest variances
for feature in ["worst area", "mean area"]:
    fig = px.box(data_frame=df, x=feature, title=f"Distribution of {feature}")
    fig.update_layout(xaxis_title=feature)
    fig.show()


# In[5]:


# Step 5: TRIMMED VARIANCES-Calculate the trimmed variances to remove extreme values and get the top 10 features
trim_var = df.apply(trimmed_var).sort_values().tail(10)
print("Top 10 trimmed variances type:", type(trim_var))
print("Top 10 trimmed variances shape:", trim_var.shape)
print(trim_var)

# Visualize the top 10 trimmed variances
fig = px.bar(
    x=trim_var,
    y=trim_var.index,
    title="Cancer: High Trimmed Variance Features"
)
fig.update_layout(xaxis_title="Trimmed Variance", yaxis_title="Feature", xaxis_tickformat='.2f')
fig.show()

# Generate a list of the top 5 features with the highest trimmed variance
high_variance_columns = trim_var.tail(5).index.to_list()
print("High variance columns type:", type(high_variance_columns))
print("High variance columns len:", len(high_variance_columns))
print(high_variance_columns)


# In[6]:


# Step 6: DATA SPLITTING-Create the feature matrix X with the top 5 features
X = df[high_variance_columns]
print("X type:", type(X))
print("X shape:", X.shape)
X.head()


# In[7]:


# Step 7: MODEL BUILDING-Standardize the data before clustering
X_summary = X.aggregate(["mean", "std"]).astype(int)
print("X_summary type:", type(X_summary))
print("X_summary shape:", X_summary.shape)
print(X_summary)

# StandardScaler transformation
ss = StandardScaler()
X_scaled_data = ss.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled_data, columns=X.columns)

print("X_scaled type:", type(X_scaled))
print("X_scaled shape:", X_scaled.shape)
X_scaled.head()

# Check the summary statistics after standardization
X_scaled_summary = X_scaled.aggregate(["mean", "std"]).astype(int)
print("X_scaled_summary type:", type(X_scaled_summary))
print("X_scaled_summary shape:", X_scaled_summary.shape)
print(X_scaled_summary)


# In[8]:


# Step 8: EVALUATE K-MEANS MODELS-Use a loop to build and evaluate K-Means models with varying cluster numbers
n_clusters = range(2, 12)
inertia_errors = []
silhouette_scores = []

for k in n_clusters:
    model = make_pipeline(StandardScaler(), KMeans(n_clusters=k, random_state=42))
    model.fit(X)
    inertia_errors.append(model.named_steps["kmeans"].inertia_)
    silhouette_scores.append(silhouette_score(X, model.named_steps["kmeans"].labels_))

print("Inertia Errors:", inertia_errors[:3])
print("Silhouette Scores:", silhouette_scores[:3])

# Plot inertia and silhouette scores
fig = px.line(x=n_clusters, y=inertia_errors, title="K-Means Model: Inertia vs Number of Clusters")
fig.update_layout(xaxis_title="Number of Clusters (k)", yaxis_title="Inertia")
fig.show()

fig = px.line(x=n_clusters, y=silhouette_scores, title="K-Means Model: Silhouette Scores vs Number of Clusters")
fig.update_layout(xaxis_title="Number of Clusters (k)", yaxis_title="Silhouette Score")
fig.show()

# Final K-Means model with the optimal number of clusters (e.g., 4)
final_model = make_pipeline(StandardScaler(), KMeans(n_clusters=4, random_state=42))
final_model.fit(X)
check_is_fitted(final_model)


# In[9]:


# Step 9: DATA COMMUNICATION-Check final model results
labels = final_model.named_steps["kmeans"].labels_
print("Labels:", labels[:5])

# Create a DataFrame with the mean values of the features for each cluster
xgb = X.groupby(labels).mean()
print("Cluster Mean Values:", xgb)

# Visualize cluster mean values
fig = px.bar(xgb, barmode="group", title="Mean Measures by Cluster")
fig.update_layout(xaxis_title="Cluster", yaxis_title="Mean")
fig.show()


# In[10]:


# Step 10: PCA IMPLEMENTATION & DIMENSIONALITY REDUCTION - Perform PCA to reduce dimensionality
pca = PCA(n_components=2, random_state=42)
X_t = pca.fit_transform(X) 
X_pca = pd.DataFrame(X_t, columns=["PC1", "PC2"])

# Display PCA DataFrame shape
print("PCA DataFrame shape:", X_pca.shape)

# Get the loading scores
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                   index=X.columns)
print("PCA Loadings:")
print(loadings)

# Plot PCA representation of clusters
fig = px.scatter(X_pca, x="PC1", y="PC2", color=labels.astype(str), title="PCA Representation of Clusters")
fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
fig.show()


# In[11]:


# Step 11:IMPLEMENT LOGISTIC REGRESSION FOR PREDICTION
# Define the features and target
features = ['mean perimeter', 'area error', 'worst perimeter', 'mean area', 'worst area']
X = df[features]
y = df['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the logistic regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Make predictions
y_pred = logistic_model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[12]:


# Step 12: SAVE THE FINAL BEST MODEL-For future predictions
import joblib

# Save the logistic regression model to a file
joblib.dump(logistic_model, 'Cancer logistic model.pkl')

print("Model saved as Cancer logistic model.pkl")


# In[ ]:




