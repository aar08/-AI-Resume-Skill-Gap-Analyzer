
# AI Resume Skill Gap Analyzer

An AI-powered web application that analyzes a candidate's resume against a job description using Natural Language Processing (NLP).

The application identifies matching skills, highlights missing skills, calculates a skill match score, and provides visual insights to help candidates improve their resumes.

## Features

- Upload resumes in PDF format
- Extract text from resumes
- Identify required skills from job descriptions
- Use semantic similarity to identify relevant experience
- Calculate an overall skill match percentage
- Display matching and missing skills
- Visualize skill coverage using interactive charts
- Show individual skill similarity scores
- Suggest learning topics based on missing skills

## Technologies Used

- Python
- Streamlit
- Sentence Transformers
- Scikit-learn
- PyPDF
- Plotly

## How It Works

1. Upload a PDF resume.
2. Enter a job description.
3. The application extracts text from the resume.
4. A pretrained Sentence Transformers model compares resume content with required skills.
5. The application calculates similarity scores and identifies matching and missing skills.
6. Results are displayed in an interactive dashboard.

## Project Structure

```text
resume-skill-analyzer/
├── app.py
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd resume-skill-analyzer
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

Activate it on macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python3 -m streamlit run app.py
```

Open the local URL displayed in your terminal to use the application.

## AI Model

This project uses the pretrained `all-MiniLM-L6-v2` model from Sentence Transformers.

The model converts text into numerical embeddings and compares their semantic similarity using cosine similarity.

The similarity threshold is configurable in the application code and should be tested against real examples before practical use.

## Limitations

- The application currently supports text-based PDF resumes.
- Skill identification uses a predefined list of skills.
- Semantic similarity does not guarantee that a candidate has a particular skill.
- The match percentage is an estimate, not a hiring recommendation.

## Future Improvements

- Extract skills automatically from a wider range of job descriptions.
- Add support for DOCX resumes.
- Generate downloadable PDF analysis reports.
- Recommend personalized learning resources.
- Deploy the application online.

## Author

Computer Engineering Student

## License

This project is available for educational and personal use.
