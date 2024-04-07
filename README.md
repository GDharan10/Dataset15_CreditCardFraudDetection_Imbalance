# Credit Card Fraud Detection Imbalance Dataset

This project focuses on detecting fraudulent credit card transactions using machine learning algorithms. The dataset used contains transactions made by European cardholders in September 2013.

## Dataset Description

The dataset contains transactions that occurred over two days, with a total of 492 frauds out of 284,807 transactions. It is highly unbalanced, with the positive class (frauds) accounting for only 0.172% of all transactions.

### Features

- **Time**: Seconds elapsed between each transaction and the first transaction in the dataset.
- **Amount**: Transaction amount.
- **V1-V28**: Numerical input variables obtained through Principal Component Analysis (PCA). Original features and additional background information are not provided due to confidentiality issues.
- **Class**: Response variable, takes value 1 in case of fraud and 0 otherwise.

## Evaluation Metric

Due to the class imbalance ratio, accuracy using the Area Under the Precision-Recall Curve (AUPRC) is recommended as the evaluation metric. Confusion matrix accuracy is not meaningful for unbalanced classification.

## Libraries Used

### Data Manipulation and Analysis
- Pandas
- NumPy
- Plotly
- Matplotlib
- Seaborn

### Preprocessing
- OrdinalEncoder

### Machine Learning
- Train-Test Split
- LazyClassifier (from lazypredict.Supervised)
- XGBClassifier (from xgboost)
- LGBMClassifier (from lightgbm)
- Metrics: top_k_accuracy_score, precision_score, recall_score

## Implementation

### Preprocessing
- PCA transformation for feature reduction.
- Scaling numerical features.

### Sampling Techniques
- Under-sampling to balance the dataset.
- Over-sampling to balance the dataset.

### Model Evaluation
- XGBClassifier and LGBMClassifier are evaluated using top_k_accuracy_score, precision_score, and recall_score.

## Results

### Without Sampling (Original Imbalanced Dataset)
- XGBClassifier: 
  - k_accuracy_score: 1.0
  - precision_score: 0.47580645161290325
  - recall_score: 0.5175438596491229

- LGBMClassifier: 
  - k_accuracy_score: 1.0
  - precision_score: 0.47540983606557374
  - recall_score: 0.5087719298245614

### With Under-Sampling
- XGBClassifier:
  - k_accuracy_score: [Insert Value]
  - precision_score: [Insert Value]
  - recall_score: [Insert Value]

- LGBMClassifier:
  - k_accuracy_score: [Insert Value]
  - precision_score: [Insert Value]
  - recall_score: [Insert Value]

### With Over-Sampling
- XGBClassifier:
  - k_accuracy_score: [Insert Value]
  - precision_score: [Insert Value]
  - recall_score: [Insert Value]

- LGBMClassifier:
  - k_accuracy_score: [Insert Value]
  - precision_score: [Insert Value]
  - recall_score: [Insert Value]

## Conclusion

- Summarize the results obtained with different sampling techniques.
- Discuss the performance of the classifiers in detecting fraudulent transactions.
- Provide insights for future improvements or considerations.

