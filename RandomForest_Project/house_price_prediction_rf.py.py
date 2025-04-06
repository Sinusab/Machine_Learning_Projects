# PART 1: Data Preprocessing
# 1) Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_log_error, make_scorer

# 2) Importing the Dataset
# Load training and test datasets from CSV files
train_df = pd.read_csv('House-Prices-Advancded-Regression/train.csv')
test_df = pd.read_csv('House-Prices-Advancded-Regression/test.csv')

# 3) Exploratory Data Analysis (EDA)
# Display dataset info, summary statistics, and initial rows
train_df.info()
test_df.info()
train_df.describe()
test_df.describe()
train_df.head()
test_df.head()

# Visualize SalePrice distribution
plt.figure(figsize=(10, 6))
sns.distplot(train_df['SalePrice'])
plt.show()

# Correlation heatmap
plt.figure(figsize=(35, 35))
sns.heatmap(train_df.corr(numeric_only=True), annot=True, fmt=".1f")
plt.show()

# Bar plots
plt.figure(figsize=(10, 6))
sns.barplot(x='YearBuilt', y='SalePrice', data=train_df)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='SaleCondition', y='SalePrice', data=train_df)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='YrSold', y='SalePrice', data=train_df)
plt.show()

# 4) Dropping Unnecessary Columns
# Remove columns with high missing values or low predictive power
train_df = train_df.drop(["Id", "Alley", "PoolQC", "Fence", "MiscFeature"], axis=1)
test_df = test_df.drop(["Alley", "PoolQC", "Fence", "MiscFeature"], axis=1)

# 5) Handling Missing Data
# Numerical columns (Train): Fill with mean or mode
train_df["LotFrontage"] = train_df["LotFrontage"].fillna(train_df["LotFrontage"].mean())
train_df["MasVnrArea"] = train_df["MasVnrArea"].fillna(train_df["MasVnrArea"].mean())

# Calculate mean and mode for GarageYrBlt
mean_garage_yr = train_df["GarageYrBlt"].mean()
mode_garage_yr = train_df["GarageYrBlt"].mode()[0]
print(f"Mean GarageYrBlt: {mean_garage_yr}")
print(f"Mode GarageYrBlt: {mode_garage_yr}")

# Create three versions of the dataset for comparison: mean, mode, and 2001
train_df_mean = train_df.copy()
train_df_mode = train_df.copy()
train_df_2001 = train_df.copy()
train_df_mean["GarageYrBlt"] = train_df_mean["GarageYrBlt"].fillna(mean_garage_yr)
train_df_mode["GarageYrBlt"] = train_df_mode["GarageYrBlt"].fillna(mode_garage_yr)
train_df_2001["GarageYrBlt"] = train_df_2001["GarageYrBlt"].fillna(2001)

# Categorical columns (Train): Fill with "None"
cat_cols_train = ("GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtFinType2", 
                  "BsmtCond", "BsmtQual", "BsmtExposure", "MasVnrType", "Electrical", 
                  "FireplaceQu", "BsmtFinType1")
for col in cat_cols_train:
    if train_df[col].dtype == "object":
        train_df_mean[col] = train_df_mean[col].fillna("None")
        train_df_mode[col] = train_df_mode[col].fillna("None")
        train_df_2001[col] = train_df_2001[col].fillna("None")

# Numerical columns (Test): Fill with mean (using mean for consistency in test)
test_df["LotFrontage"] = test_df["LotFrontage"].fillna(test_df["LotFrontage"].mean())
test_df["MasVnrArea"] = test_df["MasVnrArea"].fillna(test_df["MasVnrArea"].mean())
test_df["GarageYrBlt"] = test_df["GarageYrBlt"].fillna(mean_garage_yr)  # Using mean for test
test_df["GarageCars"] = test_df["GarageCars"].fillna(0)
test_df["GarageArea"] = test_df["GarageArea"].fillna(test_df["GarageArea"].mean())
test_df["BsmtFullBath"] = test_df["BsmtFullBath"].fillna(0)
test_df["BsmtHalfBath"] = test_df["BsmtHalfBath"].fillna(0)
test_df["BsmtFinSF1"] = test_df["BsmtFinSF1"].fillna(test_df["BsmtFinSF1"].mean())
test_df["BsmtFinSF2"] = test_df["BsmtFinSF2"].fillna(test_df["BsmtFinSF2"].mean())
test_df["TotalBsmtSF"] = test_df["TotalBsmtSF"].fillna(test_df["TotalBsmtSF"].mean())
test_df["BsmtUnfSF"] = test_df["BsmtUnfSF"].fillna(test_df["BsmtUnfSF"].mean())

# Categorical columns (Test): Fill with "None"
cat_cols_test = ("GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtFinType2", 
                 "BsmtCond", "BsmtQual", "BsmtExposure", "MasVnrType", "Electrical", 
                 "MSZoning", "Utilities", "Exterior1st", "Exterior2nd", "KitchenQual", 
                 "Functional", "FireplaceQu", "SaleType", "BsmtFinType1")
for col in cat_cols_test:
    if test_df[col].dtype == "object":
        test_df[col] = test_df[col].fillna("None")

# 6) Encoding Categorical Data with LabelEncoder
# Define categorical columns to encode
catagory_cols = ('MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
                 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 
                 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 
                 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 
                 'KitchenQual', 'Functional', 'FireplaceQu', 'PavedDrive', 'SaleType', 
                 'SaleCondition', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
                 'BsmtFinType2', 'BsmtCond', 'BsmtQual', 'BsmtExposure', 'MasVnrType', 
                 'Electrical', 'BsmtFinType1', 'ExterQual')

# Apply LabelEncoder to train datasets (mean, mode, and 2001 versions)
for c in catagory_cols:
    le = LabelEncoder()
    train_df_mean[c] = le.fit_transform(train_df_mean[c].values)
    train_df_mode[c] = le.fit_transform(train_df_mode[c].values)
    train_df_2001[c] = le.fit_transform(train_df_2001[c].values)

# Apply LabelEncoder to test dataset
for c in catagory_cols:
    le = LabelEncoder()
    test_df[c] = le.fit_transform(test_df[c].values)

# 7) Splitting the Datasets
X_train_mean = train_df_mean.drop('SalePrice', axis=1)
X_train_mode = train_df_mode.drop('SalePrice', axis=1)
X_train_2001 = train_df_2001.drop('SalePrice', axis=1)
Y_train = train_df_mean['SalePrice']
X_test = test_df.drop('Id', axis=1).copy()

# 8) Feature Scaling
sc = StandardScaler()
X_train_mean = sc.fit_transform(X_train_mean)
X_train_mode = sc.fit_transform(X_train_mode)
X_train_2001 = sc.fit_transform(X_train_2001)
X_test = sc.transform(X_test)

# 9) Dimensionality Reduction with PCA
pca = PCA(n_components=10)
X_train_mean = pca.fit_transform(X_train_mean)
X_train_mode = pca.fit_transform(X_train_mode)
X_train_2001 = pca.fit_transform(X_train_2001)
X_test = pca.transform(X_test)

# PART 2: Model Training and Evaluation
# Define RMSLE as a custom scoring function
def rmsle_score(y, y_pred):
    return np.sqrt(np.mean((np.log1p(y) - np.log1p(y_pred)) ** 2))

rmsle_scorer = make_scorer(rmsle_score, greater_is_better=False)

# Train and evaluate with RandomForest using cross-validation
regressor = RandomForestRegressor(n_estimators=200, random_state=0)

# Cross-validation for mean-filled data
scores_mean = cross_val_score(regressor, X_train_mean, Y_train, cv=5, scoring=rmsle_scorer)
rmsle_mean = -scores_mean.mean()
print(f"RMSLE with Mean: {rmsle_mean}")

# Calculate R^2 for mean-filled data
regressor.fit(X_train_mean, Y_train)
r2_mean = regressor.score(X_train_mean, Y_train)
r2_mean_percent = round(r2_mean * 100, 2)
print(f"R^2 with Mean: {r2_mean_percent}%")

# Cross-validation for mode-filled data
scores_mode = cross_val_score(regressor, X_train_mode, Y_train, cv=5, scoring=rmsle_scorer)
rmsle_mode = -scores_mode.mean()
print(f"RMSLE with Mode: {rmsle_mode}")

# Calculate R^2 for mode-filled data
regressor.fit(X_train_mode, Y_train)
r2_mode = regressor.score(X_train_mode, Y_train)
r2_mode_percent = round(r2_mode * 100, 2)
print(f"R^2 with Mode: {r2_mode_percent}%")

# Cross-validation for 2001-filled data
scores_2001 = cross_val_score(regressor, X_train_2001, Y_train, cv=5, scoring=rmsle_scorer)
rmsle_2001 = -scores_2001.mean()
print(f"RMSLE with 2001: {rmsle_2001}")

# Calculate R^2 for 2001-filled data
regressor.fit(X_train_2001, Y_train)
r2_2001 = regressor.score(X_train_2001, Y_train)
r2_2001_percent = round(r2_2001 * 100, 2)
print(f"R^2 with 2001: {r2_2001_percent}%")

# Train on full mean-filled data and predict (for submission)
regressor.fit(X_train_mean, Y_train)
Y_pred = regressor.predict(X_test)

# PART 3: Submission File
submission = pd.DataFrame({"Id": test_df["Id"], "SalePrice": Y_pred})
submission.to_csv('submission_random_forest.csv', index=False)

# PART 4: Visualization
# Scatter plot: Predicted vs Actual (on training data)
Y_pred_train = regressor.predict(X_train_mean)
plt.figure(figsize=(10, 6))
plt.scatter(Y_train, Y_pred_train, alpha=0.5)
plt.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predicted vs Actual Prices (Training Data)')
plt.show()

# Histogram of logarithmic errors
errors = np.log1p(Y_train) - np.log1p(Y_pred_train)
plt.figure(figsize=(10, 6))
sns.histplot(errors, bins=30, kde=True)
plt.xlabel('Logarithmic Error')
plt.title('Distribution of Errors')
plt.show()

# Validation split for additional evaluation
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train_mean, Y_train, test_size=0.2, random_state=0)
rf = RandomForestRegressor(n_estimators=200, random_state=0)
rf.fit(X_train_split, y_train_split)
y_pred_val = rf.predict(X_val)
rmsle_val = np.sqrt(mean_squared_log_error(y_val, y_pred_val))
print(f"RMSLE on Validation Set: {rmsle_val}")

# Scatter plot for validation set
plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred_val, alpha=0.5)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Predicted vs Actual Prices (Validation Set)')
plt.show()

# Histogram of validation errors
errors_val = np.log1p(y_val) - np.log1p(y_pred_val)
plt.figure(figsize=(10, 6))
plt.hist(errors_val, bins=30)
plt.xlabel('Logarithmic Error')
plt.title('Distribution of Validation Errors')
plt.show()