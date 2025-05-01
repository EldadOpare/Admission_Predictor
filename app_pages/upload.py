import streamlit as st
import pandas as pd
import pickle
import numpy as np
import random
from utils import preprocess_input

st.title("📚 Admission Pool Prediction")
st.markdown("Upload your CSV file and get predictions for the entire applicant pool.")

with st.expander("ℹ️ Click to view required CSV format and column details", expanded=False):
    st.markdown("""
**Please make sure your CSV contains the following columns:**

| Column Name        | Description                                       | Example                   |
|--------------------|---------------------------------------------------|---------------------------|
| `gender`           | Gender of applicant                               | Male, Female              |
| `international`    | Is the applicant international?                   | Yes, No                   |
| `gpa`              | GPA on a 4.0 scale                                | 3.75                      |
| `gmat`             | GMAT score                                        | 700                       |
| `work_exp`         | Years of work experience                          | 2                         |
| `major`            | Intended major of study                           | Business, STEM, Humanities |
| `race`             | Applicant's race/ethnic group                     | Black, Asian, White, etc. |
| `work_industry`    | Industry of work experience                       | Technology, Consulting    |
| `application_id` *(optional)* | Unique application ID (auto-generated if missing) | 123456                    |

*Note:* Column names are case-sensitive.
    """)

uploaded_file = st.file_uploader("📤 Upload CSV File", type="csv")

@st.cache_resource
def load_model():
    with open("model/final_admission_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)


        if 'applicant_id' in df.columns:
            df.rename(columns={'applicant_id': 'application_id'}, inplace=True)

        # Add application_id if missing
        if 'application_id' not in df.columns:
            df['application_id'] = [random.randint(100000, 999999) for _ in range(len(df))]

        # Convert "Yes"/"No" to boolean if necessary
        if 'international' in df.columns:
            df['international'] = df['international'].map({'Yes': True, 'No': False}).fillna(df['international'])

        processed_df = preprocess_input(df.copy())

        predictions = model.predict_proba(processed_df)[:, 1]
        df["admission_probability"] = np.round(predictions, 2)
        df["predicted_admission"] = df["admission_probability"].apply(lambda x: "Admit" if x >= 0.5 else "Reject")

        st.success("✅ Predictions completed!")
        

        # Show results in tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 All Predictions", 
            "✅ Admitted", 
            "❌ Rejected", 
            "📈 Sorted & Summary"
        ])

        with tab1:
            st.write("### All Applicants with Predictions")
            st.dataframe(df)

        with tab2:
            st.write("### Predicted Admitted")
            st.dataframe(df[df["predicted_admission"] == "Admit"])

        with tab3:
            st.write("### Predicted Rejected")
            st.dataframe(df[df["predicted_admission"] == "Reject"])

        with tab4:
            st.write("### Applicants Sorted by Admission Probability (High to Low)")
            sorted_df = df.sort_values(by="admission_probability", ascending=False).reset_index(drop=True)
            st.dataframe(sorted_df)

            total_applicants = len(df)
            admitted_count = (df["predicted_admission"] == "Admit").sum()
            rejected_count = (df["predicted_admission"] == "Reject").sum()

            st.markdown(f"""
            **Summary Statistics**  
            - Total Applicants: **{total_applicants}**  
            - Predicted Admitted: **{admitted_count}**  
            - Predicted Rejected: **{rejected_count}**  
            """)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
