import streamlit as st
import pandas as pd
import joblib

# Load preprocessor and model
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("gradient_boosting_model.pkl")

st.title("INX Future Inc Employee Performance Prediction")
st.write("Predict employee performance ratings based on employee attributes.")

st.header("Employee Information")

# Required features
features = {
     # Categorical:
    'Gender': st.selectbox("Gender", ['Male', 'Female']),
    'EducationBackground': st.selectbox("Education Background",
                                        ['Life Sciences', 'Medical', 'Marketing', 'Technical', 'Other']),
    'MaritalStatus': st.selectbox("Marital Status", ['Single', 'Married', 'Divorced']),
    'EmpDepartment': st.selectbox("Department",
                                  ['Sales', 'Research & Development', 'Human Resource','Finance','Data Science','Development']),
    'EmpJobRole': st.selectbox("Job Role",
                               ['Sales Executive','Developer','Manager','Human Resource','Research Scientist','Laboratory Technician','Healthcare Representative','Other']),
    'BusinessTravelFrequency': st.selectbox("Business Travel Frequency", ['Rarely','Frequently','Never']),
    'OverTime': st.selectbox("OverTime Work", ['Yes','No']),
    'Attrition': st.selectbox("Attrition", ['Yes','No']),
    # Numerical:
    'Age': st.number_input("Age", 18, 65, 30),
    'DistanceFromHome': st.number_input("Distance From Home in km", 0, 100, 10),
    'EmpEducationLevel': st.number_input("Education Level (1-5) With 5 being the Highest", 1, 5, 3),
    'EmpEnvironmentSatisfaction': st.selectbox("Work Environment Satisfaction", [1,2,3,4]),
    'EmpHourlyRate': st.number_input("Hourly Rate", 0, 100, 50),
    'EmpJobInvolvement': st.selectbox("Job Involvement", [1,2,3,4]),
    'EmpJobLevel': st.number_input("Job Level", 1, 10, 2),
    'EmpJobSatisfaction': st.selectbox("Job Satisfaction", [1,2,3,4]),
    'NumCompaniesWorked': st.number_input("Number of Companies Worked", 0, 20, 3),
    'EmpLastSalaryHikePercent': st.number_input("Last Salary Hike %", 0, 100, 10),
    'EmpRelationshipSatisfaction': st.selectbox("Relationship Satisfaction", [1,2,3,4]),
    'TotalWorkExperienceInYears': st.number_input("Number of Total years worked ", 0, 40, 5),
    'TrainingTimesLastYear': st.number_input("Number of Training Times Last Year", 0, 20, 2),
    'EmpWorkLifeBalance': st.selectbox("Work Life Balance", [1,2,3,4]),
    'ExperienceYearsAtThisCompany': st.number_input("Years at Current Company", 0, 40, 3),
    'ExperienceYearsInCurrentRole': st.number_input("Years in Current Role", 0, 40, 2),
    'YearsSinceLastPromotion': st.number_input("Years Since Last Promotion", 0, 20, 1),
    'YearsWithCurrManager': st.number_input("Years With Current Manager", 0, 20, 2),

   
}

# Convert to DataFrame
input_df = pd.DataFrame([features])

if st.button("Predict Performance Rating"):

    # Preprocess
    input_enc = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(input_enc)[0]

    st.success(f"Predicted Employee Performance Rating: {prediction}")
