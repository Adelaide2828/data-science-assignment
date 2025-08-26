
### **Data Science and Machine Learning Applications**

This repository contains the code and resources for a data science assignment covering predictive maintenance and fraud detection. The project demonstrates the application of machine learning algorithms, evaluation techniques, and best practices in data science workflows.

-----

### **Project Structure**

  * `data/`: Contains the datasets used for the analysis.
  * `notebooks/`: Holds the Python scripts used to perform the data analysis and modeling.
  * `output/`: Stores the generated plots and CSV files from the analysis.
  * `README.md`: This file, detailing the project's purpose and steps.

-----

### **Problem 1: Predictive Maintenance Analysis**

This section addresses a predictive maintenance problem using a **Random Forest Regressor** to predict the remaining useful life of a machine.

#### **Steps Taken:**

1.  **Environment Setup:** Installed required libraries: `pandas`, `scikit-learn`, `matplotlib`, `seaborn`.
2.  **Data Loading & Preprocessing:** Loaded the `Question 1` dataset and separated the features and the target variable (`Days to Failure`).
3.  **Model Training:** Trained a Random Forest Regressor on 80% of the data.
4.  **Model Evaluation:** Evaluated the model using **RMSE** and **R-squared** on both a test set and through **5-fold cross-validation** to assess its generalization.
5.  **Output Generation:** Saved the model's predictions to `rf_predictions.csv`.

#### **Key Results:**

  * **RMSE on Test Data:** 159.22
  * **R-squared on Test Data:** -0.20
  * **Average RMSE (Cross-Validation):** 146.74
  * **Average R-squared (Cross-Validation):** -0.26

-----

### **Problem 2: Fraud Detection in Banking**

This problem tackles a fraud detection task using a combination of unsupervised and supervised learning algorithms.

#### **Part a) Unsupervised Learning with K-Means**

1.  **Objective:** To find hidden patterns in the unlabeled transaction data.
2.  **Technique:** **K-Means Clustering** was used to group the data points.
3.  **Optimal Clusters:** The **Elbow Method** was applied to determine the optimal number of clusters, which was found to be 4.
4.  **Visualization:** The clusters were visualized using **Principal Component Analysis (PCA)** to reduce the dimensionality of the data.

#### **Part b, c, d) Supervised Learning with Naïve Bayes**

1.  **Objective:** To classify transactions as fraudulent or legitimate using the labeled data.
2.  **Model:** A **Naïve Bayes Classifier** was chosen.
3.  **Model Evaluation:** The model was evaluated using the **F1-score**, a crucial metric for handling the imbalanced nature of fraud data.
4.  **Performance:**
      * **F1-score on Test Data:** 0.14
      * **Average F1-score (Cross-Validation):** 0.17
5.  **Insights:** The model showed a high **recall** (correctly identifying the single fraudulent case) but a very low **precision** (many false alarms), as demonstrated by the **confusion matrix**.

-----

### **How to Run the Code**

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Adelaide2828/data-science-assignment.git
    ```
2.  **Install Dependencies:**
    ```bash
    pip install pandas scikit-learn matplotlib seaborn
    ```
3.  **Navigate to the `notebooks` folder:**
    ```bash
    cd data-science-assignment/notebooks
    ```
4.  **Run the scripts:**
    ```bash
    python problem1_analysis.py
    python problem2_analysis.py
    ```

