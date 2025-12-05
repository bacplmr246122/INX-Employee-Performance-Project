 # INX Future Inc. — Employee Performance Analysis & Prediction

This project provides a full machine-learning pipeline aimed at understanding and predicting employee performance at INX Future Inc. Using advanced analytics and model-driven insights, the system supports leadership in identifying key drivers of performance and improving hiring and talent-management decisions.

---

## 📌 Project Objectives
- Analyze employee-related factors to uncover patterns behind performance levels.
- Identify the top predictors influencing performance outcomes.
- Build a robust, deployable model to forecast employee performance.
- Develop a user-friendly Streamlit application for real-time predictions.

---

## 📊 Dataset Overview
The dataset includes demographic, job-related, and satisfaction-based employee attributes.  
Key columns include experience metrics, satisfaction ratings, job roles, departments, and salary-related indicators.

---

## 🛠 Methodology Summary
### **1. Data Preprocessing**
- Missing values handled using median (numeric) and constant tags (categorical).
- Outlier detection via boxplots and distribution analysis.
- OneHotEncoding used for categorical features, StandardScaler applied to numeric features.
- A ColumnTransformer automated all preprocessing steps.

### **2. Class Balancing**
Performance ratings were imbalanced (3: 874, 2: 194, 4: 132).  
SMOTENC was used to oversample minority classes while respecting categorical structure.

### **3. Modeling**
The following models were trained and evaluated:
- Decision Tree — 89.17%
- Random Forest — 91.67%
- **Gradient Boosting — 92.08% (Final choice)**

The Gradient Boosting model provided the best stability, generalization, and feature interpretability.

### **4. Deployment**
A fully interactive Streamlit app was developed:
- Clean UI for employee input
- Automatic preprocessing using saved `preprocessor.pkl`
- Performance prediction via `gradient_boosting_model.pkl`
- Real-time output for HR and hiring teams

---

## 🚀 Features of This Repository
- Complete Jupyter Notebook with EDA, preprocessing, and ML pipeline  
- Saved preprocessor and trained model files  
- Streamlit deployment script (`app.py`)  
- Documentation and report summarizing insights  

---

## 📘 Key Libraries Used
`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`,  
`imbalanced-learn (SMOTENC)`, `joblib`, `streamlit`

---

## 📄 License
This project is for educational and analytical purposes as part of the CDS certification requirements.


