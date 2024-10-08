The Breast Cancer Referral Project

The dataset analyzed in this project has been retrieved from the sklearn datasets. All the features in the dataset are in numerical form, while the target variable is in binary form, with malignant coded as 0 and benign as 1. Twelve main steps have been performed in this project, as outlined below:

Step 1: LIBRARIES LOADING - the required libraries.  

Step 2: DATA LOADING - the breast cancer dataset was loaded, which included its features and the target data.  

Step 3: DATA EXPLORATION - the variances for all features were calculated, and the top 10 features with the highest variances were identified.  

Step 4: DATA VISUALIZATION - the initial analysis of the top 10 variances was visualized using a horizontal bar chart. The boxplot of the mean area and the worst area was plotted, which showed that the two features had numerous outliers, warranting the trimming of the variances.  

Step 5: TRIMMED VARIANCES - the trimmed variances were calculated to remove extreme values and obtain the top 10 features. This was achieved by removing the top 10% and the bottom 10% from the dataset to reduce the influence of outliers, which is a robust measure of central tendency.  

Step 6: DATA SPLITTING - the feature matrices of the trimmed variances were calculated, focusing on the top 5 features with the highest variances.  

Step 7: MODEL BUILDING - this was achieved through standardization of the data before clustering, which was followed by the StandardScaler transformation.  

Step 8: EVALUATE K-MEANS MODELS - the loop code was used to build and evaluate K-Means models with varying cluster numbers, generating the inertia errors and silhouette scores. A snippet check_is_fitted (final_model) was added at the end of this code to ensure the model has been trained before attempting to use it for predictions or evaluations. The code indicates that the final model will need four clusters.  

Step 9: DATA COMMUNICATION - this code is used to check the final model results by creating a DataFrame with the mean values of the features for each cluster, whose results are visualized as comparative bars.  

Step 10: PCA IMPLEMENTATION & DIMENSIONALITY REDUCTION - principal component analysis was performed to reduce dimensionality and visualize the clusters using a scatter plot.  

Step 11: IMPLEMENT LOGISTIC REGRESSION FOR PREDICTION - the features and the target are defined and split into training and testing sets. A logistic regression model is created and fitted. Predictions are made, and the model is evaluated.  

Step 12: SAVE THE FINAL BEST MODEL - the final best model is saved for making future predictions. The saved model is named: Cancer logistic model.pkl.  

Interpretation of the PCA results  
The PCA loadings findings show the features "mean area" and "worst area" as the most influential features in the creation of the two principal components, especially PC1, and thus they are the most important features in predicting both malignant and benign cancers when it comes to the referral of cancer patients.  

Interpretation of the Logistic model 
The target variable is binary in nature, with malignant = 0 and benign = 1. The confusion matrix shows that the model rarely misclassified the cases, as it recorded 1 false negative and 2 false positives. The model demonstrated excellent performance in predicting both malignant (0) and benign (1) cases due to its high precision, recall, and F1-scores. Thus, the model is considered reliable in distinguishing between malignant and benign tumours, achieving an accuracy of 97%.  
