import streamlit as st

st.set_page_config(page_title="Ashesi Admissions Predictor", layout="centered")

st.title("🎓 Ashesi Admissions Predictor")
st.subheader("Using Machine Learning to Support Admissions Efficiency")

with st.expander("📘 Introduction"):
    st.markdown("""
    Every year, the Ashesi Admissions department processes a growing number of applicants. For instance, 
    the class of 2025 had 1,703 applicants, and by the class of 2028, this number had grown to 2,618. 
    With an average annual growth rate of 15.39%, projections show that by the class of 2032, applications 
    could reach 4,632. This exponential increase clearly indicates the need for a faster, more automated, 
    and scalable selection process to avoid bottlenecks in the current manual system.
    """)

with st.expander("❗ Problem Statement"):
    st.markdown("""
    The main issue faced by the Ashesi admissions team is the ever-increasing number of applications. 
    This surge makes it difficult to efficiently review each applicant, resulting in long processing times, 
    delayed decisions, and the possibility of missing out on qualified candidates. The pressure also places 
    a heavy burden on staff and affects the fairness and consistency of evaluations.
    """)

with st.expander("✅ Our Solution"):
    st.markdown("""
    Our team developed a machine learning-based admission prediction system using a Voting Classifier 
    that combines Logistic Regression, XGBoost, and Support Vector Machine (SVM). This model was trained 
    on publicly available admissions data to identify features that contribute to successful applications. 

    By predicting the likelihood of acceptance for each applicant, the model allows the admissions team 
    to focus their attention on borderline cases while confidently progressing with top-ranked candidates. 
    This approach enhances consistency, shortens review time, and provides scalable support for the growing 
    number of applications.
    """)

with st.expander("🛠️ Data Collection & Preprocessing Challenges"):
    st.markdown("""
    We initially sought access to Ashesi's internal admissions data but were denied due to privacy concerns. 
    To overcome this, we selected a synthetic graduate school admissions dataset from Kaggle that contained 
    attributes such as gender, race, GPA, GMAT scores, and international status.

    Preprocessing involved several key steps. Missing values in the admission column were treated as 
    rejections and labeled as "reject". Race data containing nulls were recoded as "international" to preserve 
    demographic information. Finally, one-hot encoding was applied to categorical variables to ensure compatibility 
    with machine learning algorithms like logistic regression.

    Although the data is synthetic, it served as a suitable proxy for demonstrating how ML can be leveraged 
    in the admissions process without compromising real student data.
    """)

with st.expander("⚖️ Ethical Considerations"):
    st.markdown("""
    We carefully considered data privacy and fairness during the design of our system. Since admissions data 
    often includes sensitive personal information, we deliberately avoided using any real Ashesi student data. 
    Instead, synthetic datasets were used to protect privacy.

    Additionally, we are aware that machine learning models can replicate existing biases found in training data. 
    If deployed in real admissions settings, it's essential to audit the model for fairness and actively correct 
    any biases against specific demographics. Transparency, human oversight, and ethical review must accompany 
    any real-world implementation.
    """)

with st.expander("📓 Explore Our Training Notebook"):
    st.markdown("""
    You can review the full data processing, model training, and evaluation in our Jupyter notebook hosted on Google Colab:  
    👉 [Open Notebook](https://colab.research.google.com/drive/1IHzfozZVbBNmXI2gDI5dqFZhrfyLGuxu#scrollTo=PLXeQc-HEtJP)
    """)

st.success("Use the navigation sidebar to test the prediction model with your own inputs!")
