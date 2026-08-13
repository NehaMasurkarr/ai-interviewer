import streamlit as st

st.title("AI Interviewer")

st.write("Welcome to the AI Interviewer!")

job_description = st.text_area("Paste the Job Description")

number_of_questions = st.number_input(
    "Number of interview questions",
    min_value=1,
    max_value=10,
    value=5
)

generate_button = st.button("Generate Questions")

if generate_button:
    if job_description:
        st.success("Ready to generate interview questions!")
    else:
        st.error("Please enter a job description.")