# %%
# All necessary imports and Assignment wide Constants
import pandas as pd
import numpy as np

# Preprocessing and feature engineering
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Machine learning and Deep Learning Models 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Evaluation and Visualization
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
RANDOM_STATE = 41154429 

# %% [markdown]
# # Data Understanding

# %% [markdown]
# ## 1. Collect Initial Data
# 

# %%
dataframe_original = pd.read_csv('Heart_Disease_Prediction.csv')

# Create a new index column 'ID' and Set it as the index
dataframe_original["ID"] = range(len(dataframe_original))
dataframe_original.set_index("ID", inplace=True)
dataframe_original.head()

# Check index column
print("Index column:", dataframe_original.index.name)
dataframe_original.head(300)

# %% [markdown]
# ## 2. Describe Data
# 

# %%
# Initial Statistics of the dataset
dataframe_original.describe()

# %% [markdown]
# ## 3. Explore Data
# 

# %%
# Check columns and their data types
dataframe_original.info()
print("--" * 50)
# Check for unique values in each column
print("\nUnique values per column:")
print(dataframe_original.nunique())

# %%
import matplotlib.pyplot as plt

# Plot histograms for all numeric columns with shared layout
# formula for bins = Range(max - min )/ number of bins
# layout = (rows, columns)
dataframe_original.select_dtypes(include="number").hist(
    bins=40,
    figsize=(15, 10),
    grid=True
)

plt.suptitle("Distributions of Numeric Features", fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Conversion of Class column into binary

# %%
# Convert Class 'Heart Disease' column to binary values 0/1
# Presence = 1, Absence = 0
dataframe_original['Heart Disease'] = dataframe_original['Heart Disease'].astype('category').cat.codes
# Check the unique values in 'Heart Disease' column
print("\nUnique values in 'Heart Disease' column:")
print(dataframe_original['Heart Disease'].unique())

print("\nData frame after conversion:")
dataframe_original.head(10)

# %% [markdown]
# ### Correlation analysis

# %%
# Plot Correlation Matrix 
plt.figure(figsize=(12, 8))
sns.heatmap(
    dataframe_original.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Correlation Matrix of Features", fontsize=16)
plt.tight_layout()
plt.show()

# %%
# Select Most relevant features based on correlation
relevant_features = dataframe_original.corr()['Heart Disease'].abs().sort_values(ascending=False).index[:10]
print("\nMost relevant features based on correlation with 'Heart Disease':")
print(len(relevant_features), "features")
print(relevant_features)

# Total cols with low correlation with 'Heart Disease'
low_correlation_features = dataframe_original.corr()['Heart Disease'].abs().sort_values(ascending=False).index[10:]
print("\nFeatures with low correlation with 'Heart Disease':")
print(len(low_correlation_features), "features")
print(low_correlation_features)

# %% [markdown]
# ### Check for outliers and Handle them

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Plot a boxplot for each numeric column
numeric_cols = dataframe_original.select_dtypes(include="number").columns

plt.figure(figsize=(15, 6))
dataframe_original[numeric_cols].boxplot(fontsize=10)
plt.title("Boxplot of Numeric Columns")
plt.tight_layout()
plt.show()

# %%
for col in numeric_cols:
    plt.figure(figsize=(6, 1.5))
    sns.boxplot(x=dataframe_original[col])
    plt.title(f"Boxplot of {col}")
    plt.show()


# %%
# Show distribution + Outliers using violin plot
plt.figure(figsize=(15, 6))
sns.violinplot(
    data=dataframe_original[numeric_cols],
    palette="muted",
)
plt.title("Violin Plot of Numeric Columns")
plt.tight_layout()
plt.show()

# %%
# Print first outliers for all numeric columns
print("\nSuspected Outliers in Cholesterol Column:")
print(dataframe_original['Cholesterol'].sort_values(ascending=False).head(5))
# High Cholesterol values present in range justified , not an outlier (more than 500 can be a sign of Familial Hypercholesterolemia, linked below) 
# https://my.clevelandclinic.org/health/diseases/22067-familial-hypercholesterolemia 
print("--" * 50)

print("\nSuspected Outliers in Chest pain type Column:")
print(dataframe_original['Chest pain type'].sort_values(ascending=False).head(5))
# No outliers, just a categorical column with 4 types
print("--" * 50)

print("\nSuspected Outliers in BP Column:")
print(dataframe_original['BP'].sort_values(ascending=False).head(5))
# High BP values present in dataset are justified, not outliers
print("--" * 50)

print("\nSuspected Outliers in FBS over 120:")
print(dataframe_original['FBS over 120'].sort_values(ascending=False).head(5))
# No outliers, just a categorical column with 2 types
print("--" * 50)

print("\nSuspected Outliers in Max HR")
print(dataframe_original['Max HR'].sort_values(ascending=False).head(5))
# No outliers, just a numeric column with max HR values
print("--" * 50)

print("\nSuspected Outliers in ST depression:")
print(dataframe_original['ST depression'].sort_values(ascending=False).head(5))
# No outliers
print("--" * 50)

print("\nSuspected Outliers in Number of vessels fluro:")
print(dataframe_original['Number of vessels fluro'].sort_values(ascending=False).head(5))
# No outliers

# Suspected Outliers are not removed as they are justified.


# %% [markdown]
# ## 4. Verify Data Quality

# %% [markdown]
# ### Check missing values,duplication existence and class imbalance

# %%
# Check missing values, duplication existence and class imbalance
print("Missing values in each column:")
print(dataframe_original.isnull().sum())
print("--" * 50)
# Check for duplicate rows
print("\nNumber of duplicate rows:", dataframe_original.duplicated().sum())
print("--" * 50)
# Check the distribution of the target variable (verify if class imbalance exists)
print("\nDistribution of target variable 'Heart Disease':")
print(dataframe_original["Heart Disease"].value_counts())

# %% [markdown]
# # Data Preparation

# %% [markdown]
# ## 1. Select Data

# %%
# Prepare the dataset with most relevant features
dataframe_with_most_relevant_features_after_pearson_correlation = dataframe_original.drop(columns=low_correlation_features)

# %%
# check
print("\nData frame after dropping low correlation features:")
dataframe_with_most_relevant_features_after_pearson_correlation.head()

# %% [markdown]
# ### Print basic stats like number of instances, number of attributes, first few instances
# 

# %%
# Number of columns and rows in the dataset
print("\nNumber of columns in the original dataframe :" ,dataframe_original.shape[1])
print("\nNumber of rows in the original dataframe :", dataframe_original.shape[0])

# Number of columns and rows in the dataset after dropping low correlation features
print("\nNumber of columns in the dataframe after dropping low correlation features:", dataframe_with_most_relevant_features_after_pearson_correlation.shape[1])
print("\nNumber of rows in the dataframe after dropping low correlation features:", dataframe_with_most_relevant_features_after_pearson_correlation.shape[0])

# %% [markdown]
# ## 2. Clean Data

# %%
# No cleaning needed as there are no missing values or duplicates

# %% [markdown]
# ## 3. Construct Data

# %% [markdown]
# ### Standardization

# %%
X = dataframe_with_most_relevant_features_after_pearson_correlation.drop(columns=["Heart Disease"])
y = dataframe_with_most_relevant_features_after_pearson_correlation["Heart Disease"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train) # fit + transform on training data
X_test_standardized = scaler.transform(X_test) # transform only on test data
print("\nStandardized Training Data:")
print(X_train_standardized.shape)
print("\nStandardized Testing Data:")
print(X_test_standardized.shape)

# %% [markdown]
# ## 4. Integrate Data

# %%
# No integration of any other dataset

# %% [markdown]
# ## 5. Format Data

# %% [markdown]
# ### PCA Application (Scree and Cumulative plot included)

# %%
# First Plot scree plot to visualize optimal number of components

pca_with_all_components = PCA(random_state=RANDOM_STATE)
X_train_pca_with_all_components = pca_with_all_components.fit_transform(X_train_standardized)
X_test_pca_with_all_components = pca_with_all_components.transform(X_test_standardized)

# Plot the scree plot
plt.figure(figsize=(10, 5))
plt.plot(pca_with_all_components.explained_variance_ratio_, marker='o', linestyle='-')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Scree Plot')
plt.tight_layout()
plt.show()

# For PCA with all components 
print("\nExplained Variance Ratios (all components):", pca_with_all_components.explained_variance_ratio_)
print("Sum of Explained Variance (all components):", sum(pca_with_all_components.explained_variance_ratio_))

# %%
# Second plot cumulative scree plot to visualize optimal number of components

cumulative_variance = np.cumsum(pca_with_all_components.explained_variance_ratio_)

plt.figure(figsize=(10, 5))
plt.plot(cumulative_variance, marker='o', linestyle='-', color='green')

# Add labels to each point
for i, value in enumerate(cumulative_variance):
    plt.text(i, value, f"{value:.2f}", fontsize=9, ha='right', va='bottom')
# Optimal threshold line (targeting 80% cumulative variance)
plt.axhline(y=0.8, color='r', linestyle=':')

# Labels and grid
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Scree Plot')
plt.grid(True)
plt.tight_layout()
plt.show()

# Print Cumulative Variance Ratios for each component
for i, value in enumerate(cumulative_variance, start=1):
    print(f"Component {i} → {value:.4f}")

# %%
# D is the optimal number of components based on the cumulative variance and explained variance ratio
D = 5 

# Apply PCA with the optimal number of components
pca_with_optimal_components = PCA(n_components=D,random_state=RANDOM_STATE)
X_train_pca = pca_with_optimal_components.fit_transform(X_train_standardized)
X_test_pca = pca_with_optimal_components.transform(X_test_standardized)

# Print the explained variance ratio for the optimal number of components
print("\nExplained Variance Ratios (optimal components):", pca_with_optimal_components.explained_variance_ratio_)
print("Sum of Explained Variance (optimal components):", sum(pca_with_optimal_components.explained_variance_ratio_))

# %% [markdown]
# ### LDA Application

# %%
# Apply LDA for dimensionality reduction
lda = LDA(n_components=1)  # LDA can have at most (number of classes - 1) components
X_train_lda = lda.fit_transform(X_train_standardized, y_train) # Needs y_train class labels for fitting and transforming
X_test_lda = lda.transform(X_test_standardized)

# %%
# The dataset has 10 features after feature selection.
# LDA projects these 10-dimensional feature vectors into 1D by finding a weight vector (w).
# Each sample's 1D LDA value is the dot product of w and its 10 features.
# This projection maximizes class separation by placing samples on a line.

# The x-axis in the plot shows these 1D projected values (scores).
# The y-axis shows the estimated probability density of samples for each class at each LDA value.
# This density is computed using Kernel Density Estimation (KDE), which smooths the distribution
# of points to show how likely samples of each class occur near each position on the LDA axis.
# The area under each class's density curve sums to 1, representing the full distribution of that class.
# Peaks in the density curve indicate regions where many samples of that class are concentrated.
# This helps visualize how well LDA separates the classes along this 1D axis.

# The decision boundary in LDA is learned by finding a threshold along the 1D LDA axis
# that best separates the classes based on their projected values.
# After projecting all samples onto the LDA component, LDA estimates the class distributions
# (usually assuming Gaussian distributions) along this axis.
# The boundary is placed where the likelihood of belonging to either class is equal,
# minimizing misclassification.
# In 2D or higher dimensions, this corresponds to a linear hyperplane,
# but since we have 1D projections here, it’s just a single cutoff value on the LDA axis.

plt.figure(figsize=(15, 6))

# Plot KDE for each class (train set)
sns.kdeplot(X_train_lda[y_train == 0, 0], label="Absence (class 0)")
sns.kdeplot(X_train_lda[y_train == 1, 0], label="Presence (class 1)")

# Calculate means of the projected data for each class
mean_0 = X_train_lda[y_train == 0, 0].mean()
mean_1 = X_train_lda[y_train == 1, 0].mean()
# Plot decision boundary
decision_boundary = (mean_0 + mean_1) / 2

plt.axvline(x=decision_boundary, color='red', linestyle='--', label='Decision Boundary')
plt.xlabel("LDA Component 1")
plt.ylabel("Density")
plt.title("LDA 1D Projection - Train Set")
plt.legend()
plt.show()


# %% [markdown]
# # Modeling & Evaluation

# %% [markdown]
# ### Modeling Function

# %%
# Dictionaries to store trained models as per the training and testing sets
trained_models = {}
trained_models_PCA = {}
trained_models_LDA = {}

# Three sets of Dataframes
Dataset_standardized = "std"
Dataset_PCA = "pca"
Dataset_LDA = "lda"


def train_and_evaluate_models(Dataset, Algorithm, Algorithm_params=None):
    """
    Train a machine learning model and evaluate its performance on the selected dataset.

    Parameters:
    -----------
    Dataset : str
        Type of dataset to use for training. Options:
        - 'std': Standardized dataset
        - 'pca': PCA-transformed dataset
        - 'lda': LDA-transformed dataset
    
    Algorithm : class
        The machine learning algorithm class to use (e.g., LogisticRegression, SVC)
    
    Algorithm_params : dict, optional
        Dictionary of parameters to initialize the algorithm (default: None)

    Returns:
    --------
    model : object
        Trained model instance
    
    metrics : dict
        Dictionary containing evaluation metrics:
        - 'Accuracy': Model accuracy score
        - 'Precision': Precision score
        - 'Recall': Recall score
        - 'F1 Score': F1 score
        - 'Confusion Matrix': Confusion matrix array
    """

    if Algorithm_params is None:
        Algorithm_params = {}
    
    # 3 sets of Dataframes
    if Dataset == "std":
        x_train_set, x_test_set, y_train_set, y_test_set = X_train_standardized, X_test_standardized, y_train, y_test
    elif Dataset == "pca":
        x_train_set, x_test_set, y_train_set, y_test_set = X_train_pca, X_test_pca, y_train, y_test
    elif Dataset == "lda":
        x_train_set, x_test_set, y_train_set, y_test_set = X_train_lda, X_test_lda, y_train, y_test
    else: 
        raise ValueError("Invalid dataset type. Choose from 'std', 'pca', or 'lda'.")
    

    # Train
    model = Algorithm(**Algorithm_params)
    model.fit(x_train_set, y_train_set)

    print(f"\nTraining and evaluating : {Algorithm.__name__} with parameters {Algorithm_params} for {Dataset}")

    # Make predictions
    y_pred = model.predict(x_test_set)

    # Evaluate the model
    accuracy = accuracy_score(y_test_set, y_pred)
    precision = precision_score(y_test_set, y_pred)
    recall = recall_score(y_test_set, y_pred)
    f1 = f1_score(y_test_set, y_pred)
    conf_matrix = confusion_matrix(y_test_set, y_pred)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
    }
    print("Conf Matrix:\n", conf_matrix)
    return model,metrics

def run_evaluations(Dataset, Algorithm, Algorithm_param_sets=None, store_trained_models=None, model_label_prefix=None):
    """
    Run model evaluations with different parameter sets and store results.

    Parameters:
    -----------
    Dataset : str
        Type of dataset to use ('std', 'pca', or 'lda')
    
    Algorithm : class
        The machine learning algorithm class to use
    
    Algorithm_param_sets : list of tuples, optional
        List of (set_name, params_dict) tuples containing parameter sets
        
    store_trained_models : dict, optional
        Dictionary to store trained model instances
        
    model_label_prefix : str, optional
        Prefix to use for model names when storing

    Returns:
    --------
    None (prints results and stores trained models if specified)
    """
    results = {}

    for set_name, Algorithm_params in Algorithm_param_sets:
        model, metrics = train_and_evaluate_models(Dataset, Algorithm, Algorithm_params)

        # Save metrics
        results[set_name] = metrics

        # Save trained models in respective dictionaries
        if store_trained_models is not None:
            full_model_name = f"{model_label_prefix}_{set_name}" if model_label_prefix else set_name
            store_trained_models[full_model_name] = model
            
    results_df = pd.DataFrame(results)
    print("\nResults:")
    print(results_df)
    return None

# %% [markdown]
# ### 1. Random Forest, 2. MLP, 3. SVM, 4. Logistic Regression, 5. kNN

# %%
# Constants for Controlling the models

ALGORITHM_1 = KNeighborsClassifier
ALGORITHM_1_PARAMS_set_1 = {'n_neighbors': 5, 'metric': 'minkowski', 'p': 2}
ALGORITHM_1_PARAMS_set_2 = {'n_neighbors': 10, 'metric': 'cosine', 'weights': 'distance'}
ALGORITHM_1_PARAMS_set_3 = {'n_neighbors': 10, 'metric': 'manhattan', 'weights': 'uniform'}
param_sets_KNN = [('Set 1', ALGORITHM_1_PARAMS_set_1), ('Set 2', ALGORITHM_1_PARAMS_set_2), ('Set 3', ALGORITHM_1_PARAMS_set_3)]

ALGORITHM_2 = LogisticRegression
ALGORITHM_2_PARAMS_set_1 = {'random_state': RANDOM_STATE, 'solver': 'liblinear', 'max_iter': 100, 'C': 1.0}
ALGORITHM_2_PARAMS_set_2 = {'random_state': RANDOM_STATE, 'solver': 'saga', 'max_iter': 200, 'C': 0.5, 'penalty': 'elasticnet', 'l1_ratio': 0.5}
ALGORITHM_2_PARAMS_set_3 = {'random_state': RANDOM_STATE, 'solver': 'newton-cg', 'max_iter': 300, 'C': 0.1}
param_sets_LR = [('Set 1', ALGORITHM_2_PARAMS_set_1), ('Set 2', ALGORITHM_2_PARAMS_set_2), ('Set 3', ALGORITHM_2_PARAMS_set_3)]

ALGORITHM_3 = RandomForestClassifier
ALGORITHM_3_PARAMS_set_1 = {'random_state': RANDOM_STATE, 'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 2}
ALGORITHM_3_PARAMS_set_2 = {'random_state': RANDOM_STATE, 'n_estimators': 200, 'max_depth': 10 , 'min_samples_split': 10}
ALGORITHM_3_PARAMS_set_3 = {'random_state': RANDOM_STATE, 'n_estimators': 50, 'max_depth': 3, 'min_samples_split': 5}
param_sets_RF = [('Set 1', ALGORITHM_3_PARAMS_set_1), ('Set 2', ALGORITHM_3_PARAMS_set_2), ('Set 3', ALGORITHM_3_PARAMS_set_3)]

ALGORITHM_4 = SVC
ALGORITHM_4_PARAMS_set_1 = {'kernel': 'rbf','C': 1.0,'gamma': 'scale'}
ALGORITHM_4_PARAMS_set_2 = {'kernel': 'poly','C': 1.0, 'gamma': 'auto' ,'degree': 3}
ALGORITHM_4_PARAMS_set_3 = {'kernel': 'sigmoid','C': 1.0,'gamma': 'scale'}
param_sets_SVC = [('Set 1', ALGORITHM_4_PARAMS_set_1), ('Set 2', ALGORITHM_4_PARAMS_set_2), ('Set 3', ALGORITHM_4_PARAMS_set_3)]

ALGORITHM_5 = MLPClassifier
ALGORITHM_5_PARAMS_set_1 = {'random_state': RANDOM_STATE,'hidden_layer_sizes': (100, 50),'activation': 'relu','solver': 'adam','max_iter': 500,}
ALGORITHM_5_PARAMS_set_2 = {'random_state': RANDOM_STATE,'hidden_layer_sizes': (50, 25),'activation': 'tanh','solver': 'sgd','max_iter': 300,}
ALGORITHM_5_PARAMS_set_3 = {'random_state': RANDOM_STATE,'hidden_layer_sizes': (200, 100),'activation': 'logistic','solver': 'adam','max_iter': 1000,}
param_sets_MLP = [('Set 1', ALGORITHM_5_PARAMS_set_1), ('Set 2', ALGORITHM_5_PARAMS_set_2), ('Set 3', ALGORITHM_5_PARAMS_set_3)]

# %% [markdown]
# ### 3 sets of Dataframe:-
# 1. Standardized Dataset
# 2. After PCA
# 3. After LDA

# %%
# Use the function to train and evaluate models 

##################################################################################
# 1. Train and evaluate models using the standardized training and testing sets
##################################################################################

# 1. K-Nearest Neighbors
run_evaluations(Dataset=Dataset_standardized, Algorithm=ALGORITHM_1, Algorithm_param_sets=param_sets_KNN, store_trained_models=trained_models, model_label_prefix="KNN")
print("\n" + "-" * 50)

# 2. Logistic Regression
run_evaluations(Dataset=Dataset_standardized, Algorithm=ALGORITHM_2, Algorithm_param_sets=param_sets_LR, store_trained_models=trained_models, model_label_prefix="LR")
print("\n" + "-" * 50)

# 3. Random Forest Classifier
run_evaluations(Dataset=Dataset_standardized, Algorithm=ALGORITHM_3, Algorithm_param_sets=param_sets_RF, store_trained_models=trained_models, model_label_prefix="RF")
print("\n" + "-" * 50)

# # 4. Support Vector Classifier
run_evaluations(Dataset=Dataset_standardized, Algorithm=ALGORITHM_4, Algorithm_param_sets=param_sets_SVC, store_trained_models=trained_models, model_label_prefix="SVC")
print("\n" + "-" * 50)

# 5. Multi-layer Perceptron Classifier
run_evaluations(Dataset=Dataset_standardized, Algorithm=ALGORITHM_5, Algorithm_param_sets=param_sets_MLP, store_trained_models=trained_models, model_label_prefix="MLP")

# %%
##################################################################################
# 2. Train and evaluate models using the PCA transformed training and testing sets
##################################################################################

# 1. K-Nearest Neighbors
run_evaluations(Dataset=Dataset_PCA, Algorithm=ALGORITHM_1, Algorithm_param_sets=param_sets_KNN, store_trained_models=trained_models_PCA, model_label_prefix="KNN")
print("\n" + "-" * 50)

# 2. Logistic Regression
run_evaluations(Dataset=Dataset_PCA, Algorithm=ALGORITHM_2, Algorithm_param_sets=param_sets_LR, store_trained_models=trained_models_PCA, model_label_prefix="LR")
print("\n" + "-" * 50)

# 3. Random Forest Classifier
run_evaluations(Dataset=Dataset_PCA, Algorithm=ALGORITHM_3, Algorithm_param_sets=param_sets_RF, store_trained_models=trained_models_PCA, model_label_prefix="RF")
print("\n" + "-" * 50)

# 4. Support Vector Classifier
run_evaluations(Dataset=Dataset_PCA, Algorithm=ALGORITHM_4, Algorithm_param_sets=param_sets_SVC, store_trained_models=trained_models_PCA, model_label_prefix="SVC")
print("\n" + "-" * 50)

# 5. Multi-layer Perceptron Classifier
run_evaluations(Dataset=Dataset_PCA, Algorithm=ALGORITHM_5, Algorithm_param_sets=param_sets_MLP, store_trained_models=trained_models_PCA, model_label_prefix="MLP")


# %%
##################################################################################
# 3. Train and evaluate models using the LDA transformed training and testing sets
##################################################################################
# 1. K-Nearest Neighbors
run_evaluations(Dataset=Dataset_LDA, Algorithm=ALGORITHM_1, Algorithm_param_sets=param_sets_KNN, store_trained_models=trained_models_LDA, model_label_prefix="KNN")
print("\n" + "-" * 50)

# 2. Logistic Regression
run_evaluations(Dataset=Dataset_LDA, Algorithm=ALGORITHM_2, Algorithm_param_sets=param_sets_LR, store_trained_models=trained_models_LDA, model_label_prefix="LR")
print("\n" + "-" * 50)

# 3. Random Forest Classifier
run_evaluations(Dataset=Dataset_LDA, Algorithm=ALGORITHM_3, Algorithm_param_sets=param_sets_RF, store_trained_models=trained_models_LDA, model_label_prefix="RF")
print("\n" + "-" * 50)

# 4. Support Vector Classifier
run_evaluations(Dataset=Dataset_LDA, Algorithm=ALGORITHM_4, Algorithm_param_sets=param_sets_SVC, store_trained_models=trained_models_LDA, model_label_prefix="SVC")
print("\n" + "-" * 50)

# 5. Multi-layer Perceptron Classifier
run_evaluations(Dataset=Dataset_LDA, Algorithm=ALGORITHM_5, Algorithm_param_sets=param_sets_MLP, store_trained_models=trained_models_LDA, model_label_prefix="MLP")

# %% [markdown]
# ### BEST SETS AND EVALUATION 

# %%
# knn:
# std: set 2
# pca: set 1
# lda: set 2

# lr:
# std: set 3
# pca: set 3
# lda: set 3 

# rf:
# std: set 3
# pca: set 3
# lda: set 3

# svm:
# std: set 3
# pca: set 3
# lda: set 1

# mlp:
# std: set 3
# pca: set 3
# lda: set 3

# Use the trained models (best sets as per evaluated performance) to instantiate the predictions and plot the results
# Best models PCA
best_model_KNN = trained_models_PCA["KNN_Set 1"]
best_model_LR = trained_models_PCA["LR_Set 3"]
best_model_RF = trained_models_PCA["RF_Set 3"]
best_model_SVC = trained_models_PCA["SVC_Set 3"]
best_model_MLP = trained_models_PCA["MLP_Set 3"]

# Best models LDA
best_model_KNN_LDA = trained_models_LDA["KNN_Set 2"]
best_model_LR_LDA = trained_models_LDA["LR_Set 3"]
best_model_RF_LDA = trained_models_LDA["RF_Set 3"]
best_model_SVC_LDA = trained_models_LDA["SVC_Set 1"]
best_model_MLP_LDA = trained_models_LDA["MLP_Set 3"]

# Predict PCA
y_pred_best_KNN_PCA = best_model_KNN.predict(X_test_pca)
y_pred_best_LR_PCA = best_model_LR.predict(X_test_pca)
y_pred_best_RF_PCA = best_model_RF.predict(X_test_pca)
y_pred_best_SVC_PCA = best_model_SVC.predict(X_test_pca)
y_pred_best_MLP_PCA = best_model_MLP.predict(X_test_pca)

# Predict LDA
y_pred_best_KNN_LDA = best_model_KNN_LDA.predict(X_test_lda)
y_pred_best_LR_LDA = best_model_LR_LDA.predict(X_test_lda)
y_pred_best_RF_LDA = best_model_RF_LDA.predict(X_test_lda)
y_pred_best_SVC_LDA = best_model_SVC_LDA.predict(X_test_lda)
y_pred_best_MLP_LDA = best_model_MLP_LDA.predict(X_test_lda)


# %%
def plot_pca_test_actual_vs_predicted(pca_testing_set, actual_set_testing, predicted_set_testing):
    # Misclassified points
    misclassified = actual_set_testing != predicted_set_testing

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Actual classes plot
    axs[0].scatter(pca_testing_set[:, 0], pca_testing_set[:, 1], c=actual_set_testing,
                   cmap='coolwarm', edgecolor='k', alpha=0.8)
    axs[0].set_title('Actual Classes (Test Set)')

    # Plot 2: Predicted classes plot
    axs[1].scatter(pca_testing_set[:, 0], pca_testing_set[:, 1], c=predicted_set_testing,
                   cmap='coolwarm', edgecolor='k', alpha=0.8)

    # Outline misclassified points with different colors per true class
    unique_classes = np.unique(y_test)
    outline_colors = ['blue', 'red']  

    for cls, color in zip(unique_classes, outline_colors):
        cls_misclassified = misclassified & (y_test == cls)
        axs[1].scatter(pca_testing_set[cls_misclassified, 0], pca_testing_set[cls_misclassified, 1],
                       facecolors='none', edgecolors=color, s=200, linewidths=2, 
                       label=f'Misclassified Class {cls}')

    axs[1].set_title('Predicted Classes (Test Set)')

    # Axis labels
    for ax in axs:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    # Descriptive class labels
    class_label_map = {
        0: '0 -> Heart Disease = Absence',
        1: '1 -> Heart Disease = Presence'
    }

    class_labels = np.unique(np.concatenate((y_test, predicted_set_testing)))
    class_handles = [
        mpatches.Patch(color=plt.cm.coolwarm(i / (len(class_labels) - 1)), 
                       label=class_label_map.get(label, f'Class {label}'))
        for i, label in enumerate(class_labels)
    ]

    # Legend for misclassified points
    misclass_handles = [
        mpatches.Patch(edgecolor=color, facecolor='none', label=f'Misclassified Class {cls}')
        for cls, color in zip(unique_classes, outline_colors)
    ]

    axs[0].legend(handles=class_handles, loc='upper left')
    axs[1].legend(handles=class_handles + misclass_handles, loc='upper left')

    # Print Confusion Matrix
    conf_matrix = confusion_matrix(actual_set_testing, predicted_set_testing)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    plt.suptitle("PCA: Actual vs Predicted (Test Set)")
    plt.tight_layout()
    plt.show()

# %%
def plot_pca_train_actual_and_support_vectors(pca_training_set, actual_set_training, support_vectors_pca, support_vector_labels):
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Actual classes in training set
    axs[0].scatter(pca_training_set[:, 0], pca_training_set[:, 1], c=actual_set_training, 
                   cmap='coolwarm', edgecolor='k', alpha=0.8)
    axs[0].set_title('Actual Classes (Train Set)')

    # Plot 2: Training set with support vectors
    axs[1].scatter(pca_training_set[:, 0], pca_training_set[:, 1], c=actual_set_training,
                   cmap='coolwarm', edgecolor='k', alpha=1.0, label='Training Data')

    # Plot support vectors with different outline colors per class
    unique_sv_labels = np.unique(support_vector_labels)
    outline_colors = ['blue', 'red']  

    for cls, color in zip(unique_sv_labels, outline_colors):
        idx = support_vector_labels == cls
        axs[1].scatter(support_vectors_pca[idx, 0], support_vectors_pca[idx, 1],
                       facecolors='none', edgecolors=color, s=200, linewidths=2,
                       marker='o', label=f'Support Vectors Class {cls}')

    # Axis labels
    for ax in axs:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    # Descriptive class labels
    class_label_map = {
        0: '0 -> Heart Disease = Absence',
        1: '1 -> Heart Disease = Presence'
    }

    class_labels = np.unique(y_train)
    class_handles = [
        mpatches.Patch(color=plt.cm.coolwarm(i / (len(class_labels) - 1)),
                       label=class_label_map.get(label, f'Class {label}'))
        for i, label in enumerate(class_labels)
    ]

    # Legend handles for support vectors per class
    sv_handles = [
        mpatches.Patch(edgecolor=color, facecolor='none', label=f'Support Vectors Class {cls}')
        for cls, color in zip(unique_sv_labels, outline_colors)
    ]

    axs[0].legend(handles=class_handles, loc='upper left')
    axs[1].legend(handles=class_handles + sv_handles, loc='upper left')

    plt.suptitle("PCA: Train Set with Support Vectors")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### After applying PCA, actual classes and predicted classes of the test set, misclassified instances should be circled with a different color (2 plots in parallel)

# %%
# Plotting implementation for PCA results
print("\nPlotting PCA Results...")

# 1. After PCA, KNN : Actual vs Predicted on Test Set
print("\nPlotting Actual vs Predicted for KNN after PCA")
plot_pca_test_actual_vs_predicted(X_test_pca, y_test, y_pred_best_KNN_PCA)

# 2. After PCA, LR : Actual vs Predicted on Test Set
print("\nPlotting Actual vs Predicted for LR after PCA")
plot_pca_test_actual_vs_predicted(X_test_pca, y_test, y_pred_best_LR_PCA)

# 3. After PCA, RF : Actual vs Predicted on Test Set
print("\nPlotting Actual vs Predicted for RF after PCA")
plot_pca_test_actual_vs_predicted(X_test_pca, y_test, y_pred_best_RF_PCA)

# 4. After PCA, SVC : Actual vs Predicted on Test Set
print("\nPlotting Actual vs Predicted for SVC after PCA")
plot_pca_test_actual_vs_predicted(X_test_pca, y_test, y_pred_best_SVC_PCA)

# 5. After PCA, MLP : Actual vs Predicted on Test Set
print("\nPlotting Actual vs Predicted for MLP after PCA")
plot_pca_test_actual_vs_predicted(X_test_pca, y_test, y_pred_best_MLP_PCA)

# %% [markdown]
# ### After applying PCA, actual classes of the train set color coded by class and then in the second plot, support vectors should also be plotted, color coded by class (2 plots in parallel)

# %%
support_vectors_pca = best_model_SVC.support_vectors_
support_vector_indices = best_model_SVC.support_
support_vector_labels = y_train.iloc[support_vector_indices]

plot_pca_train_actual_and_support_vectors(X_train_pca, y_train, support_vectors_pca, support_vector_labels)


# %% [markdown]
# ### After applying LDA, actual classes and predicted classes of the test set, misclassified instances should be circled with a different color (2 plots in parallel)

# %%
def plot_lda_actual_vs_predicted(lda_testing_set, actual_testing_set, predicted_testing_set, RANDOM_STATE=42):
    range_lda = np.max(lda_testing_set[:, 0]) - np.min(lda_testing_set[:, 0])
    jitter_strength = 0.05 * range_lda  # 5% jitter

    unique_classes = np.unique(actual_testing_set)
    misclassified = actual_testing_set != predicted_testing_set

    outline_colors = ['orange', 'blue']

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    np.random.seed(RANDOM_STATE)
    jitter_all = np.random.uniform(-jitter_strength, jitter_strength, size=lda_testing_set.shape[0])

    # To store scatter plot handles per class for legend colors
    scatter_handles_actual = {}
    scatter_handles_pred = {}

    for ax, data, title in zip(axs, [actual_testing_set, predicted_testing_set], ['Actual Classes', 'Predicted Classes']):
        for cls in unique_classes:
            cls_mask = (data == cls)
            cls_points = lda_testing_set[cls_mask, 0]
            cls_jitter = jitter_all[cls_mask]

            scatter = ax.scatter(cls_points, cls_jitter, label=f'Class {cls}', alpha=0.7, edgecolor='k')
            # Save scatter handle to get color later
            if title == 'Actual Classes':
                scatter_handles_actual[cls] = scatter
            else:
                scatter_handles_pred[cls] = scatter

        if title == 'Predicted Classes':
            mis_points = lda_testing_set[misclassified, 0]
            mis_jitter = jitter_all[misclassified]

            for pred_cls in unique_classes:
                mask_pred_cls = (predicted_testing_set[misclassified] == pred_cls)
                ax.scatter(
                    mis_points[mask_pred_cls], mis_jitter[mask_pred_cls],
                    facecolors='none',
                    edgecolors=outline_colors[pred_cls % len(outline_colors)],
                    s=200,
                    linewidths=2,
                    label=f'Misclassified Class {pred_cls}'
                )

        ax.set_yticks([])
        ax.set_xlabel("LDA Component 1")
        ax.set_title(title)
    
    # Descriptive class labels
    class_label_map = {
        0: '0 -> Heart Disease = Absence',
        1: '1 -> Heart Disease = Presence'
    }
    # Extract colors from scatter plots for actual classes
    class_handles_actual = [
        mpatches.Patch(color=scatter_handles_actual[cls].get_facecolors()[0],
                       label=class_label_map.get(cls, f'Class {cls}'))
        for cls in unique_classes
    ]

    # For predicted plot (second), legend includes classes + misclassified outlines
    class_handles_pred = [
        mpatches.Patch(color=scatter_handles_pred[cls].get_facecolors()[0],
                       label=class_label_map.get(cls, f'Class {cls}'))
        for cls in unique_classes
    ]

    misclassified_handles = [
        mpatches.Patch(edgecolor=color, facecolor='none', label=f'Misclassified Class {cls}')
        for cls, color in zip(unique_classes, outline_colors)
    ]

    axs[0].legend(handles=class_handles_actual, loc='upper left')
    axs[1].legend(handles=class_handles_pred + misclassified_handles, loc='upper left')

    conf_matrix = confusion_matrix(actual_testing_set, predicted_testing_set)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    plt.tight_layout()
    plt.show()


# %%
# Plotting implementation for LDA results
print("\nPlotting LDA Results...")

# 1. After LDA, KNN : Actual vs Predicted on Test Set
print("\nPlotting Actual vs Predicted for KNN after LDA")
plot_lda_actual_vs_predicted(X_test_lda, y_test, y_pred_best_KNN_LDA)

# 2. After LDA, LR : Actual vs Predicted on Test Set
print("\nPlotting Actual vs Predicted for LR after LDA")
plot_lda_actual_vs_predicted(X_test_lda, y_test, y_pred_best_LR_LDA)

# 3. After LDA, RF : Actual vs Predicted on Test Set
print("\nPlotting Actual vs Predicted for RF after LDA")
plot_lda_actual_vs_predicted(X_test_lda, y_test, y_pred_best_RF_LDA)

# 4. After LDA, SVC : Actual vs Predicted on Test Set
print("\nPlotting Actual vs Predicted for SVC after LDA")
plot_lda_actual_vs_predicted(X_test_lda, y_test, y_pred_best_SVC_LDA)

# 5. After LDA, MLP : Actual vs Predicted on Test Set
print("\nPlotting Actual vs Predicted for MLP after LDA")
plot_lda_actual_vs_predicted(X_test_lda, y_test, y_pred_best_MLP_LDA)

# %% [markdown]
# ### After applying LDA, actual classes of the train set color coded by class and then in the second plot, support vectors should also be plotted, color coded by class (2 plots in parallel)

# %%
def plot_lda_train_actual_and_support_vectors(lda_training_set, actual_set_training,
                                              support_vectors_lda, support_vector_labels):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Compute adaptive jitter
    range_lda = np.max(lda_training_set[:, 0]) - np.min(lda_training_set[:, 0])
    jitter_strength = 0.05 * range_lda

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Actual classes in training set
    y_jitter = np.random.uniform(-jitter_strength, jitter_strength, size=lda_training_set.shape[0])

    for cls in np.unique(actual_set_training):
        idx = actual_set_training == cls
        axs[0].scatter(lda_training_set[idx, 0], y_jitter[idx],
                       label=f'Class {cls}', alpha=0.7, edgecolor='k')

    axs[0].set_title('Actual Classes (Train Set)')
    axs[0].set_xlabel('LDA Component 1')
    axs[0].set_yticks([])

    # Plot 2: Training set with support vectors
    for cls in np.unique(actual_set_training):
        idx = actual_set_training == cls
        axs[1].scatter(lda_training_set[idx, 0], y_jitter[idx],
                       label=f'Class {cls}', alpha=0.7, edgecolor='k')

    # Match support vectors to training set to get the same jitter
    support_colors = ['lime', 'darkgreen']
    for cls, color in zip(np.unique(support_vector_labels), support_colors):
        sv_idx = support_vector_labels == cls
        sv_points = support_vectors_lda[sv_idx, 0]

        # Match each support vector to training data
        for x in sv_points:
            match = np.where(np.isclose(lda_training_set[:, 0], x, atol=1e-6))[0]
            if len(match) > 0:
                j = y_jitter[match[0]]
            else:
                j = np.random.uniform(-jitter_strength, jitter_strength)
            axs[1].scatter([x], [j], facecolors='none', edgecolors=color,
                           s=200, linewidths=2, label=f'Support Vectors Class {cls}')

    axs[1].set_title('Train Set with Support Vectors')
    axs[1].set_xlabel('LDA Component 1')
    axs[1].set_yticks([])

    # Legends
    class_label_map = {
        0: '0 -> Heart Disease = Absence',
        1: '1 -> Heart Disease = Presence'
    }

    class_handles = [
        mpatches.Patch(color='gray', label=class_label_map.get(cls, f'Class {cls}'))
        for cls in np.unique(actual_set_training)
    ]

    sv_handles = [
        mpatches.Patch(edgecolor=color, facecolor='none', label=f'Support Vectors Class {cls}')
        for cls, color in zip(np.unique(support_vector_labels), support_colors)
    ]

    axs[0].legend(handles=class_handles)
    axs[1].legend(handles=class_handles + sv_handles)

    plt.suptitle("LDA: Train Set with Support Vectors")
    plt.tight_layout()
    plt.show()


# %%
support_vectors_lda = best_model_SVC_LDA.support_vectors_
support_vector_indices = best_model_SVC_LDA.support_
support_vector_labels = y_train.iloc[support_vector_indices]
print("\nPlotting Train Set with Support Vectors for LDA")
plot_lda_train_actual_and_support_vectors(X_train_lda, y_train, support_vectors_lda, support_vector_labels)


