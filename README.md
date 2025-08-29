# Loan Approval Prediction – Machine Learning Project

## Project Overview

Developed a machine learning model to predict loan approval status based on applicant demographics, financial details, and credit history. The project involved exploratory data analysis (EDA), data preprocessing, feature engineering, and the application of classification algorithms to build a robust prediction model.

## Key Steps & Methodology

### Data Exploration & Visualization
- Performed univariate, bivariate, and multivariate analysis using Seaborn & Matplotlib.
- Identified trends in applicant income, loan amount, and credit history affecting loan approval.
- Visualized categorical distributions (gender, marital status, education, property area).

### Data Cleaning & Preprocessing
- Handled missing values using mode and mean imputation.
- Applied one-hot encoding for categorical variables.
- Addressed outliers using IQR method.
- Scaled features with MinMaxScaler.

### Class Imbalance Handling
- Applied SMOTE (Synthetic Minority Oversampling Technique) to balance approved vs. rejected loans.

### Feature Engineering
- Performed log transformations on skewed variables (ApplicantIncome, CoapplicantIncome, LoanAmount).
- Created correlation heatmaps to evaluate relationships among numerical variables.

### Model Building & Evaluation
- Implemented Logistic Regression and K-Nearest Neighbors (KNN) classifiers.
- Optimized KNN by testing multiple values of k (1–18) to maximize accuracy.
- Evaluated performance using confusion matrix, classification report, and accuracy scores.

## Tools & Technologies
- Languages/Libraries: Python, Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, Imbalanced-learn
- Techniques: EDA, Outlier Treatment, SMOTE, One-Hot Encoding, Feature Scaling, Logistic Regression, KNN
- Environment: Google Colab

## Results & Insights
- Achieved high classification accuracy with Logistic Regression and optimized KNN.
- Credit history, applicant income, and loan amount were found to be the most influential features.
- Visual insights confirmed that applicants with a positive credit history had a much higher loan approval rate.
