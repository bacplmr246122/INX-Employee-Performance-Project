import streamlit as st
import pandas as pd
import joblib

# Load preprocessor and model
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("gradient_boosting_model.pkl")

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="INX Employee Performance Predictor",
    page_icon="📊",
    layout="centered"
)

# ------------------------------
# HEADER SECTION
# ------------------------------
st.title("📈 INX Future Inc – Employee Performance Prediction")
st.markdown(
    """
    Welcome to the **Employee Performance Prediction App**!  
    Provide employee attributes below and the model will predict their **Performance Rating**.  
    """
)

st.markdown("---")

# ------------------------------
# EMPLOYEE INFO SECTION
# ------------------------------
st.header("🧑‍💼 Employee Information")

features = {
     # ------------------------------
     # CATEGORICAL FIELDS
     # ------------------------------
    'Gender': st.selectbox("🚻 Gender", ['Male', 'Female']),
    
    'EducationBackground': st.selectbox(
        "🎓 Education Background",
        ['Life Sciences', 'Medical', 'Marketing', 'Technical', 'Other']
    ),
    
    'MaritalStatus': st.selectbox(
        "💍 Marital Status",
        ['Single', 'Married', 'Divorced']
    ),
    
    'EmpDepartment': st.selectbox(
        "🏢 Department",
        ['Sales', 'Research & Development', 'Human Resource','Finance','Data Science','Development']
    ),
    
    'EmpJobRole': st.selectbox(
        "👔 Job Role",
        ['Sales Executive','Developer','Manager','Human Resource',
         'Research Scientist','Laboratory Technician','Healthcare Representative','Other']
    ),
    
    'BusinessTravelFrequency': st.selectbox(
        "✈ Business Travel Frequency",
        ['Rarely','Frequently','Never']
    ),
    
    'OverTime': st.selectbox("⏱ OverTime Work", ['Yes','No']),
    
    'Attrition': st.selectbox("📉 Attrition", ['Yes','No']),

    # ------------------------------
    # NUMERIC FIELDS
    # ------------------------------
    'Age': st.number_input("🎯 Age", 18, 65, 30),
    'DistanceFromHome': st.number_input("📍 Distance From Home (km)", 0, 100, 10),
    'EmpEducationLevel': st.number_input("🎓 Education Level (1–5)", 1, 5, 3),
    'EmpEnvironmentSatisfaction': st.selectbox("🌿 Environment Satisfaction", [1,2,3,4]),
    'EmpHourlyRate': st.number_input("💰 Hourly Rate", 0, 100, 50),
    'EmpJobInvolvement': st.selectbox("📌 Job Involvement", [1,2,3,4]),
    'EmpJobLevel': st.number_input("📊 Job Level (1–10)", 1, 10, 2),
    'EmpJobSatisfaction': st.selectbox("😊 Job Satisfaction", [1,2,3,4]),
    'NumCompaniesWorked': st.number_input("🏢 Companies Worked", 0, 20, 3),
    'EmpLastSalaryHikePercent': st.number_input("📈 Last Salary Hike (%)", 0, 100, 10),
    'EmpRelationshipSatisfaction': st.selectbox("🤝 Relationship Satisfaction", [1,2,3,4]),
    'TotalWorkExperienceInYears': st.number_input("🧠 Total Work Experience (yrs)", 0, 40, 5),
    'TrainingTimesLastYear': st.number_input("📘 Trainings Last Year", 0, 20, 2),
    'EmpWorkLifeBalance': st.selectbox("⚖ Work–Life Balance", [1,2,3,4]),
    'ExperienceYearsAtThisCompany': st.number_input("🏢 Years at Company", 0, 40, 3),
    'ExperienceYearsInCurrentRole': st.number_input("👔 Years in Current Role", 0, 40, 2),
    'YearsSinceLastPromotion': st.number_input("🚀 Years Since Last Promotion", 0, 20, 1),
    'YearsWithCurrManager': st.number_input("🧑‍🤝‍🧑 Years With Current Manager", 0, 20, 2),
}

# Convert to DataFrame
input_df = pd.DataFrame([features])

st.markdown("---")

# ------------------------------
# PREDICTION SECTION
# ------------------------------
st.subheader("🔮 Predict Employee Performance")

if st.button("✨ Predict Performance Rating"):

    # Preprocess
    input_enc = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(input_enc)[0]

    # Display result
    st.success(f"🏆 **Predicted Employee Performance Rating: {prediction}**")

    # Nice message
    if prediction == 4:
        st.info("🌟 This employee shows very high performance potential!")
    elif prediction == 3:
        st.info("👍 This employee has solid and consistent performance.")
    elif prediction == 2:
        st.warning("⚠ Employee may require performance improvement support.")

st.markdown("---")
st.markdown("Developed using Streamlit")


