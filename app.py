import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy # type: ignore
import pdfplumber

# Load the SpaCy model and sentence transformer model
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to extract text from PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to extract skills (simple example)
def extract_skills(text):
    skills = set()
    doc = nlp(text)
    for token in doc:
        if token.pos_ == "NOUN" and len(token.text) > 3:
            skills.add(token.text.lower())
    return skills

# Function to parse contact info
def parse_resume_fields(text):
    doc = nlp(text)
    parsed_fields = {'Name': '', 'Email': '', 'Phone': ''}
    for ent in doc.ents:
        if ent.label_ == 'PERSON' and not parsed_fields['Name']:
            parsed_fields['Name'] = ent.text
        elif ent.label_ == 'EMAIL' and not parsed_fields['Email']:
            parsed_fields['Email'] = ent.text
        elif ent.label_ == 'PHONE' and not parsed_fields['Phone']:
            parsed_fields['Phone'] = ent.text
    return parsed_fields

# Streamlit interface
st.title("Resume Matcher")

# Sidebar for job description and skills
uploaded_files = st.file_uploader("Upload Resumes (PDF or TXT)", accept_multiple_files=True, type=['pdf', 'txt'])
job_description_file = st.file_uploader("Upload Job Description (TXT)", type='txt')

# Upload job description
if job_description_file:
    jd_text = job_description_file.read().decode('utf-8')
    job_embedding = model.encode([jd_text])[0]
    jd_skills = extract_skills(jd_text)

    # Match resumes to job description
    results = []
    if uploaded_files:
        with st.spinner('🔎 Matching resumes, please wait...'):
            for file in uploaded_files:
                resume_text = extract_text_from_pdf(file) if file.name.endswith('.pdf') else file.read().decode('utf-8')
                resume_embedding = model.encode([resume_text])[0]
                base_score = cosine_similarity([resume_embedding], [job_embedding])[0][0]

                resume_skills = extract_skills(resume_text)
                matched_skills = resume_skills.intersection(jd_skills)

                boost = 0
                for crit_skill in jd_skills:
                    if crit_skill in resume_skills:
                        boost += 0.05

                final_score = min(base_score + boost, 1.0)

                parsed_fields = parse_resume_fields(resume_text)
                highlighted_resume = resume_text  # Add any additional highlight logic here

                results.append({
                    'Resume_Name': file.name,
                    'Name': parsed_fields['Name'],
                    'Email': parsed_fields['Email'],
                    'Phone': parsed_fields['Phone'],
                    'Match_Percentage': round(final_score * 100, 2),
                    'Matched_Skills': ', '.join(matched_skills),
                    'Highlighted_Resume': highlighted_resume
                })

        # Show results
        results_df = pd.DataFrame(results)
        st.subheader("🏆 Resume Matching Results")
        st.dataframe(results_df[['Resume_Name', 'Name', 'Email', 'Phone', 'Match_Percentage', 'Matched_Skills']])

        # Show highlighted resume
        selected_resume = st.selectbox("🔍 View Highlighted Resume:", results_df['Resume_Name'])
        if selected_resume:
            resume_row = results_df[results_df['Resume_Name'] == selected_resume].iloc[0]
            st.subheader(f"📄 {resume_row['Name']} - Highlighted Skills for {resume_row['Resume_Name']}")
            st.markdown(resume_row['Highlighted_Resume'], unsafe_allow_html=True)
