# House Price Prediction with RandomForest

This project uses the RandomForestRegressor algorithm to predict house prices based on the "House Prices: Advanced Regression Techniques" dataset from Kaggle. The code includes data preprocessing, model training, evaluation using RMSLE and R^2, and visualization of results.

## Features
- Data preprocessing: Handling missing values, encoding categorical variables, feature scaling, and PCA.
- Model: RandomForestRegressor with 200 trees.
- Evaluation metrics: RMSLE (Root Mean Squared Logarithmic Error) and R^2 score.
- Visualization: Scatter plots and error histograms.

## Requirements
To run this project, install the required Python libraries listed in `requirements.txt`.

## How to Run
1. Ensure you have the dataset files (`train.csv` and `test.csv`) in the `House-Prices-Advanced-Regression/` directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
