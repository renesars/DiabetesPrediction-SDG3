# Diabetes Prediction Using Machine Learning
Welcome to my Diabetes Prediction project! 🎉 This project aims to predict whether an individual has diabetes based on a set of health features using various machine learning models. The goal is to help healthcare professionals identify potential diabetes patients early, which can significantly improve diagnosis and treatment outcomes.

## 💡 About the Project
Diabetes is one of the most prevalent health conditions, and early detection can help prevent complications. In this project, I’ve used machine learning techniques to predict whether a person is likely to develop diabetes based on medical and lifestyle features.

### The Problem:
The challenge is to predict whether a person is diabetic or not using a dataset that contains various health-related metrics such as age, glucose levels, BMI, and blood pressure. The problem is a **binary classification**, where the output is either "diabetic" or "non-diabetic."

### Why I Chose This:
I chose this project because diabetes is a growing concern globally, and machine learning has immense potential in transforming healthcare. By predicting diabetes accurately, we can provide better preventative care and improve quality of life for patients.

---

## 🚀 How to Run This Project

1. **Clone the Repository:**

   First, clone the repository to your local machine using:

   ```bash
   git clone https://github.com/renesars/diabetes-prediction.git
   ```

2. **Install Dependencies:**

   The project requires several Python libraries. You can install all dependencies using `pip` from the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

3. **Running the Code:**

   The main code for running the diabetes prediction is in the **`diabetes_prediction.ipynb`** Jupyter notebook. Simply open the notebook and run the cells in order. The notebook will walk you through:

   - Data preprocessing
   - Model training and evaluation (using Random Forest, Logistic Regression, and SVM)
   - Model comparison using ROC curves
   - Cross-validation for enhanced model evaluation

---

## 📊 What’s Inside the Project?

1. **Data Preprocessing:**
   - The dataset has been cleaned and preprocessed by handling missing values, scaling features, and addressing class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
  
2. **Model Building:**
   - **Random Forest:** A robust model that works well for classification tasks.
   - **Logistic Regression:** A simple but effective linear model for binary classification.
   - **Support Vector Machine (SVM):** A powerful model that works well in high-dimensional spaces.
  
3. **Model Evaluation:**
   - Accuracy, Precision, Recall, F1-score, and ROC-AUC are used to evaluate each model.
   - ROC curves are plotted to visually compare the models’ performances.

4. **K-Fold Cross-Validation:**
   - Cross-validation (CV) is performed to assess how well the models generalize to unseen data. This step is essential for avoiding overfitting.

5. **Visualization:**
   - **ROC Curves**: Visual comparison of model performance.
   - **AUC Scores**: Quantitative measure to understand model effectiveness.

---

## 🤖 How I Built This

### **1. Data Collection and Preprocessing:**
I used the Pima Indians Diabetes Dataset, which contains several health features and a binary outcome indicating whether a person is diabetic. I made sure to handle missing values, normalize the data, and perform feature engineering where necessary.

### **2. Model Selection:**
I experimented with three machine learning models:
- **Random Forest:** Chosen for its robustness and ability to handle imbalanced data.
- **Logistic Regression:** A good baseline model for binary classification.
- **Support Vector Machine (SVM):** Selected for its capability in handling complex decision boundaries.

### **3. Evaluation and Cross-Validation:**
I used a combination of accuracy, precision, recall, F1-score, and ROC-AUC to evaluate the models. To make sure that the models weren’t overfitting, I also performed K-fold cross-validation.

---

## 📈 Results

After training and evaluating all three models, I compared them based on their performance metrics and plotted the **ROC curves**. Here’s a quick snapshot of the evaluation:

- **Random Forest**: High accuracy and robust performance across all metrics.
- **Logistic Regression**: Good baseline performance, but slightly less accurate than Random Forest.
- **SVM**: Performed well, especially in cases of complex decision boundaries.

---

## 🔧 Technologies Used

- **Python:** For data analysis and machine learning.
- **Pandas & NumPy:** For data manipulation and preprocessing.
- **Scikit-learn:** For building and evaluating machine learning models.
- **Imbalanced-learn (SMOTE):** For handling class imbalance.
- **Matplotlib:** For plotting the ROC curves and visualizations.
- **Jupyter Notebook:** For interactive coding and step-by-step execution.

---

## 📄 Files in this Repository

- **`diabetes_prediction.ipynb`**: Main Jupyter notebook containing the code for the project.
- **`requirements.txt`**: List of dependencies needed to run the project.
- **`diabetes.csv`**: Folder containing the dataset (if applicable).
- **`README.md`**: This file with detailed information about the project.

---

## 🙋‍♂️ Acknowledgments
I’d like to thank the developers of the Pima Indians Diabetes Dataset and the creators of the various Python libraries I used. Special thanks to the open-source community for making this project possible!

---

## 📢 Final Thoughts
This project has been an exciting journey of applying machine learning to real-world problems. The next steps could include exploring more complex models, implementing feature selection techniques, and deploying the model as a web application for real-time predictions. 

If you have any questions or suggestions, feel free to reach out!

---
