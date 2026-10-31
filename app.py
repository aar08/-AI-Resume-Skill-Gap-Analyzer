
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import re

# Page setup
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("AI Resume Skill Gap Analyzer")
st.write("Analyze your resume against a job description using NLP.")

# Skill descriptions help the AI understand related concepts
skills = {
    "Python": "Programming using Python language",
    "Java": "Java programming and object oriented development",
    "C++": "C++ programming and software development",
    "SQL": "Database queries and SQL database management",
    "Machine Learning": "Building predictive models using machine learning algorithms",
    "Data Analysis": "Analyzing data using statistics and data processing",
    "Pandas": "Data manipulation and analysis using the Pandas library",
    "NumPy": "Numerical computing using NumPy arrays",
    "Flask": "Building web applications using Flask",
    "Streamlit": "Building interactive Python data applications",
    "HTML": "Creating web pages using HTML",
    "CSS": "Styling web pages using CSS",
    "JavaScript": "Programming interactive web applications",
    "Git": "Version control using Git",
    "GitHub": "Hosting code repositories and collaborating using GitHub",
    "Deep Learning": "Neural networks and deep learning techniques",
    "TensorFlow": "Building machine learning models using TensorFlow",
    "Scikit-learn": "Machine learning tools and algorithms using scikit-learn",
    "Power BI": "Business intelligence and interactive data visualization",
    "Excel": "Spreadsheet analysis and calculations using Microsoft Excel",
    "MongoDB": "Working with NoSQL document databases",
    "Communication": "Communicating ideas clearly with others",
    "Problem Solving": "Analyzing problems and developing effective solutions"
}

# Load the pretrained NLP model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Input fields
resume_file = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste the job description",
    height=200
)

# Analyze button
if st.button("Analyze Resume", type="primary"):

    if resume_file is None or not job_description.strip():
        st.warning("Please upload a resume and enter a job description.")

    else:
        try:
            # Extract text from PDF
            reader = PdfReader(resume_file)

            resume_text = " ".join(
                page.extract_text() or ""
                for page in reader.pages
            ).lower()

            job_text = job_description.lower()

            if not resume_text.strip():
                st.error("No readable text found in the PDF.")
                st.stop()

            # Identify skills mentioned in the job description
            required_skills = [
                skill for skill in skills
                if re.search(
                    r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)",
                    job_text
                )
            ]

            if not required_skills:
                st.warning(
                    "No recognized skills found in the job description. "
                    "Try including skills such as Python, SQL, or Machine Learning."
                )
                st.stop()

            # Split resume into sentences
            resume_sentences = [
                sentence.strip()
                for sentence in re.split(r"[.!?\n]+", resume_text)
                if sentence.strip()
            ]

            # Load NLP model
            with st.spinner("Analyzing resume with AI..."):
                model = load_model()

                # Convert resume sentences into embeddings
                resume_embeddings = model.encode(
                    resume_sentences,
                    normalize_embeddings=True
                )

                matched_skills = []
                missing_skills = []
                skill_scores = {}

                for skill in required_skills:

                    # Create a description of the required skill
                    skill_text = skill + ": " + skills[skill]

                    skill_embedding = model.encode(
                        [skill_text],
                        normalize_embeddings=True
                    )

                    # Compare skill meaning with resume sentences
                    similarities = cosine_similarity(
                        skill_embedding,
                        resume_embeddings
                    )[0]

                    score = float(similarities.max()) * 100

                    skill_scores[skill] = round(score, 1)

                    # Exact matches count as matches
                    exact_match = re.search(
                        r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)",
                        resume_text
                    )

                    # Semantic threshold for related experience
                    if exact_match or score >= 45:
                        matched_skills.append(skill)
                    else:
                        missing_skills.append(skill)

            # Calculate overall score
            match_percentage = (
                len(matched_skills) / len(required_skills)
            ) * 100

            st.subheader("Analysis Results")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Skill Match",
                f"{match_percentage:.1f}%"
            )

            col2.metric(
                "Matching Skills",
                len(matched_skills)
            )

            col3.metric(
                "Missing Skills",
                len(missing_skills)
            )

            # Visualize skill match
            chart_data = {
                "Category": ["Matched", "Missing"],
                "Skills": [
                    len(matched_skills),
                    len(missing_skills)
                ]
            }

            fig = px.pie(
                chart_data,
                names="Category",
                values="Skills",
                title="Required Skill Coverage",
                hole=0.4
            )

            st.plotly_chart(fig, use_container_width=True)

            # Display matched skills
            st.subheader("Matching Skills")

            if matched_skills:
                st.write(", ".join(matched_skills))
            else:
                st.write("No matching skills found.")

            # Display missing skills
            st.subheader("Skills to Improve")

            if missing_skills:
                for skill in missing_skills:
                    st.write(f"- Learn {skill}")
            else:
                st.write("No missing skills found.")

            # Display individual similarity scores
            st.subheader("Skill Similarity Scores")

            score_data = {
                "Skill": list(skill_scores.keys()),
                "Similarity": list(skill_scores.values())
            }

            score_fig = px.bar(
                score_data,
                x="Skill",
                y="Similarity",
                title="Similarity by Skill",
                range_y=[0, 100]
            )

            st.plotly_chart(score_fig, use_container_width=True)

            st.caption(
                "This is an experimental AI-based skill comparison. "
                "Similarity scores do not guarantee job suitability."
            )

        except Exception as e:
            st.error(f"Unable to analyze the resume: {e}")
