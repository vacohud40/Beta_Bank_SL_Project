# Predicting Customer Churn at Beta Bank

## Project Description

Beta Bank, like many financial institutions, faces the challenge of customer churn, where clients discontinue their services at a concerning rate. Retaining existing customers is significantly more cost-effective than acquiring new ones. This project addresses this challenge by developing a predictive machine learning model capable of accurately identifying customers who are highly likely to churn (leave the bank) in the near future. This proactive identification will enable Beta Bank to implement targeted retention strategies.

## Analysis Goals

The primary goal of this project is to build a robust classification model for customer churn prediction. This will involve:

* **Data Exploration and Preprocessing:** Understanding the features related to customer behavior and contract termination, handling missing values, encoding categorical variables, and scaling numerical features.
* **Model Training and Selection:** Experimenting with various classification algorithms, including Logistic Regression and Random Forest Classifier, and potentially using a `DummyClassifier` as a baseline for performance comparison. Hyperparameter tuning will be conducted to optimize model performance.
* **Performance Evaluation:** Rigorously evaluating the models' performance using key metrics:
    * **F1 Score:** The primary metric, with a target of **at least 0.59** on the test set, balancing precision and recall, which is crucial for imbalanced datasets common in churn prediction.
    * **AUC-ROC Score:** To assess the model's ability to discriminate between positive and negative classes across various classification thresholds.
* **Strategic Insights:** Providing insights into factors contributing to churn and recommending potential retention strategies based on model findings.

## Technologies Used

* **Python**
* **Pandas:** For efficient data loading, manipulation, and analysis.
* **NumPy:** For numerical operations.
* **Scikit-learn:** For comprehensive machine learning functionalities including:
    * `train_test_split`: Data splitting.
    * `StandardScaler`, `OneHotEncoder`: Data preprocessing.
    * `LogisticRegression`: Baseline classification model.
    * `RandomForestClassifier`: Ensemble classification model.
    * `DummyClassifier`: For establishing a baseline performance.
    * `f1_score`, `roc_auc_score`, `roc_curve`, `accuracy_score`: Model evaluation metrics.
    * `GridSearchCV`: Hyperparameter tuning.
    * `shuffle`: For data shuffling (potentially for handling class imbalance or cross-validation).
* **Matplotlib:** For creating static data visualizations, such as ROC curves and feature importance plots.
* **Seaborn:** For producing aesthetically pleasing and informative statistical graphics.
* **Jupyter Notebook:** The primary environment for conducting the analysis, experimentation, and presenting findings.

## Dataset

The project utilizes historical customer data from Beta Bank, containing information on customer behavior and contract terminations.

* **File Name:** `Churn.csv`
* **Content:** This dataset includes features related to customer demographics (e.g., gender, age), account information (e.g., balance, number of products), and transaction history, along with a target variable indicating whether the customer churned.
* **Original Location (on training platform):** `/datasets/Churn.csv`

