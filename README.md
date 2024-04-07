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
  - precision_score : 0.956989247311828
  - recall_score : 0.7235772357723578

- LGBMClassifier: 
  - k_accuracy_score: 1.0
  - precision_score : 0.2049469964664311
  - recall_score : 0.4715447154471545

### With Under-Sampling
- XGBClassifier:
  - k_accuracy_score : 1.0
  - precision_score : 0.47580645161290325
  - recall_score : 0.5175438596491229

- LGBMClassifier:
  - k_accuracy_score : 1.0
  - precision_score : 0.47540983606557374
  - recall_score : 0.5087719298245614

### With Over-Sampling
- XGBClassifier:
  - k_accuracy_score : 1.0
  - precision_score : 0.9819819819819819
  - recall_score : 0.956140350877193

- LGBMClassifier:
  - k_accuracy_score : 1.0
  - precision_score : 1.0
  - recall_score : 0.9473684210526315

## Conclusion

In this credit card fraud detection project, we experimented with different sampling techniques and evaluated the performance of XGBClassifier and LGBMClassifier in detecting fraudulent transactions.

### Without Sampling (Original Imbalanced Dataset)
- Both XGBClassifier and LGBMClassifier achieved perfect k_accuracy_score (1.0) on the original imbalanced dataset. However, the precision and recall scores varied significantly between the two models. 
  - XGBClassifier achieved a high precision score (0.957) but a relatively lower recall score (0.724), indicating that while it correctly identifies many fraudulent transactions, it also misclassifies some legitimate transactions as fraudulent.
  - LGBMClassifier, on the other hand, showed a lower precision score (0.205) but a higher recall score (0.472), suggesting that it identifies a higher proportion of actual fraudulent transactions but also flags a larger number of legitimate transactions incorrectly.

### With Under-Sampling
- Both XGBClassifier and LGBMClassifier achieved perfect k_accuracy_score (1.0) with under-sampling. The precision and recall scores improved slightly compared to the original imbalanced dataset, but they were still relatively balanced. However, the performance did not surpass that of the models trained on the original imbalanced dataset.

### With Over-Sampling
- Applying over-sampling techniques significantly improved the performance of both classifiers. Both XGBClassifier and LGBMClassifier achieved perfect k_accuracy_score (1.0), indicating a perfect match between predicted and actual labels. Moreover, the precision and recall scores substantially increased, especially for LGBMClassifier, which achieved a perfect precision score of 1.0 while maintaining a high recall score (0.947).

### Performance Discussion
- Over-sampling techniques, particularly when applied in conjunction with LGBMClassifier, demonstrated the best performance in detecting fraudulent transactions. These techniques effectively balanced the dataset and improved the precision and recall scores, ensuring a higher detection rate of actual fraud cases while minimizing false positives.

