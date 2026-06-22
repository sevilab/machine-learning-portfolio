Customer Churn Analysis Project

This project focuses on analyzing **customer churn** for an e-commerce platform and building a predictive model to identify customers likely to leave. The goal is to help the business reduce churn and improve customer retention strategies.

Project Overview

* Objective:*Identify patterns in customer behavior and predict churn using machine learning.
* Dataset: Contains customer demographic, transactional, and app usage data.

Key Columns:

* CustomerID– Unique customer identifier
* Churn– Target variable (1 = churned, 0 = retained)
* Gender – Customer gender
* Tenure– Duration of customer relationship
* HourSpendOnApp– Average hours spent on app
* OrderCount – Number of orders placed
* OrderAmountHikeFromLastYear – Order amount increase from last year
* PreferredPaymentMode – Payment method preference
* PreferredLoginDevice – Device used to login
* WarehouseToHome – Delivery distance
* CouponUsed – Number of coupons used
* PreferedOrderCat – Preferred product category
* MaritalStatus– Marital status

Analysis Steps

1. Data Cleaning & Preprocessing

   * Missing values filled using median/mean
   * Categorical variables encoded using LabelEncoder

2. Exploratory Data Analysis (EDA)

   * Distribution plots for numeric features
   * Count plots for categorical features vs churn
   * Correlation heatmap to identify relationships
     
3. Feature Scaling & Train-Test Split

   * StandardScaler applied to numerical features
   * Stratified 70-30 split for training and testing sets

4. Modeling

   * Random Forest Classifier used to predict churn
   * Model trained on scaled training data

5. Evaluation

   * Confusion matrix visualized with heatmap
   * Classification report including precision, recall, F1-score
   * ROC curve plotted and ROC AUC score calculated
   * Feature importances visualized to understand key drivers of churn

 Key Insights

* Gender, preferred payment mode, and app usage metrics significantly influence churn.
* Customers with low tenure or fewer orders are more likely to churn.
* Top features driving churn can guide targeted retention strategies.

 Libraries Used
pandas, matplotlib, seaborn– Data manipulation and visualization
scikit-learn – Preprocessing, model building, and evaluation

How to Run

1. Install required libraries:
pip install pandas matplotlib seaborn scikit-learn
2. Load dataset and update file path in the script:
python
file_path = 'path_to/E Commerce Dataset.xlsx'

3. Run the Python script to perform preprocessing, modeling, and generate evaluation charts.

 Outcome

* Predictive model identifies customers at risk of churn
* Visualizations and metrics provide actionable insights for retention strategies
* Supports data-driven decision-making for marketing and customer engagement

