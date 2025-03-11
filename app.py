from flask import Flask, request, jsonify
import pandas as pd
import neattext.functions as nfx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from flask_cors import CORS
import requests
import ffmpeg
import whisper
import tempfile
from transformers import pipeline
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
import re
import pickle
from io import BytesIO
import numpy as np
import spacy

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Load dataset for course recommendations
DATASET_PATH = "./user_course_data_test.csv"
df = pd.read_csv(DATASET_PATH)

# Preprocess course data
df['clean_course_name'] = df['course_name'].apply(nfx.remove_stopwords)
df['clean_course_name'] = df['clean_course_name'].apply(nfx.remove_special_characters)

# Combine features for course recommendations
df['combined_features'] = (
    df['clean_course_name'] + ' ' +
    df['course_level'].fillna('') + ' ' +
    df['rating'].astype(str) + ' ' +
    df['exam_category'].fillna('') + ' ' +
    df['exam_score'].astype(str)
)

# Vectorization for course recommendations
tfidf = TfidfVectorizer(stop_words='english')
tfidf_mat = tfidf.fit_transform(df['combined_features'])

# Cosine similarity matrix for course recommendations
cosine_sim_mat = cosine_similarity(tfidf_mat)

# Course index mapping for course recommendations
course_index = pd.Series(df.index, index=df['course_name']).drop_duplicates()

# Load Whisper model for audio transcription
model = whisper.load_model("base")

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load models for resume parsing
rf_classifier_categorization = pickle.load(open('models/rf_classifier_categorization.pkl', 'rb'))
tfidf_vectorizer_categorization = pickle.load(open('models/tfidf_vectorizer_categorization.pkl', 'rb'))
rf_classifier_job_recommendation = pickle.load(open('models/rf_classifier_job_recommendation.pkl', 'rb'))
tfidf_vectorizer_job_recommendation = pickle.load(open('models/tfidf_vectorizer_job_recommendation.pkl', 'rb'))

# Load spaCy model for name extraction
nlp = spacy.load("en_core_web_sm")

# Load Longformer model for resume categorization
classifier = pipeline("text-classification", model="allenai/longformer-base-4096")

# Clean resume text
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

# Extract section from resume text
def extract_section(text, section_title):
    pattern = rf"{section_title}:(.*?)(?:\n[A-Z]|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""

# Predict category using Longformer
def predict_category(resume_text):
    skills_section = extract_section(resume_text, "Skills")
    education_section = extract_section(resume_text, "Education")
    resume_text = cleanResume(skills_section + " " + education_section)
    result = classifier(resume_text)[0]
    predicted_label = result['label']
    label_mapping = {
        "LABEL_0": "Software Engineer",
        "LABEL_1": "Data Scientist",
        "LABEL_2": "Web Developer",
        "LABEL_3": "DevOps Engineer",
    }
    predicted_category = label_mapping.get(predicted_label, "Unknown Category")
    return predicted_category

# Job recommendation
def job_recommendation(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    recommended_job = rf_classifier_job_recommendation.predict(resume_tfidf)[0]
    return recommended_job

# Extract text from PDF
def pdf_to_text(file):
    reader = PdfReader(file)
    text = ''
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

# Extract contact number from resume
def extract_contact_number_from_resume(text):
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    return match.group() if match else None

# Extract email from resume
def extract_email_from_resume(text):
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    return match.group() if match else None

# Extract skills from resume
def extract_skills_from_resume(text):
    skills_list = [
        'Python', 'Data Analysis', 'Machine Learning', 'Communication', 'Project Management', 'Deep Learning', 'SQL',
        'Tableau', 'Java', 'C++', 'JavaScript', 'HTML', 'CSS', 'React', 'Angular', 'Node.js', 'MongoDB', 'Express.js', 'Git',
        'Research', 'Statistics', 'Quantitative Analysis', 'Qualitative Analysis', 'SPSS', 'R', 'Data Visualization',
        'Matplotlib', 'Seaborn', 'Plotly', 'Pandas', 'Numpy', 'Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'NLTK', 'Text Mining',
        'Natural Language Processing', 'Computer Vision', 'Image Processing', 'OCR', 'Speech Recognition', 'Recommendation Systems',
        'Collaborative Filtering', 'Content-Based Filtering', 'Reinforcement Learning', 'Neural Networks', 'Convolutional Neural Networks',
        'Recurrent Neural Networks', 'Generative Adversarial Networks', 'XGBoost', 'Random Forest', 'Decision Trees', 'Support Vector Machines',
        'Linear Regression', 'Logistic Regression', 'K-Means Clustering', 'Hierarchical Clustering', 'DBSCAN', 'Association Rule Learning',
        'Apache Hadoop', 'Apache Spark', 'MapReduce', 'Hive', 'HBase', 'Apache Kafka', 'Data Warehousing', 'ETL', 'Big Data Analytics',
        'Cloud Computing', 'Amazon Web Services (AWS)', 'Microsoft Azure', 'Google Cloud Platform (GCP)', 'Docker', 'Kubernetes', 'Linux',
        'Shell Scripting', 'Cybersecurity', 'Network Security', 'Penetration Testing', 'Firewalls', 'Encryption', 'Malware Analysis',
        'Digital Forensics', 'CI/CD', 'DevOps', 'Agile Methodology', 'Scrum', 'Kanban', 'Continuous Integration', 'Continuous Deployment',
        'Software Development', 'Web Development', 'Mobile Development', 'Backend Development', 'Frontend Development', 'Full-Stack Development',
        'UI/UX Design', 'Responsive Design', 'Wireframing', 'Prototyping', 'User Testing', 'Adobe Creative Suite', 'Photoshop', 'Illustrator',
        'InDesign', 'Figma', 'Sketch', 'Zeplin', 'InVision', 'Product Management', 'Market Research', 'Customer Development', 'Lean Startup',
        'Business Development', 'Sales', 'Marketing', 'Content Marketing', 'Social Media Marketing', 'Email Marketing', 'SEO', 'SEM', 'PPC',
        'Google Analytics', 'Facebook Ads', 'LinkedIn Ads', 'Lead Generation', 'Customer Relationship Management (CRM)', 'Salesforce',
        'HubSpot', 'Zendesk', 'Intercom', 'Customer Support', 'Technical Support', 'Troubleshooting', 'Ticketing Systems', 'ServiceNow',
        'ITIL', 'Quality Assurance', 'Manual Testing', 'Automated Testing', 'Selenium', 'JUnit', 'Load Testing', 'Performance Testing',
        'Regression Testing', 'Black Box Testing', 'White Box Testing', 'API Testing', 'Mobile Testing', 'Usability Testing', 'Accessibility Testing',
        'Cross-Browser Testing', 'Agile Testing', 'User Acceptance Testing', 'Software Documentation', 'Technical Writing', 'Copywriting',
        'Editing', 'Proofreading', 'Content Management Systems (CMS)', 'WordPress', 'Joomla', 'Drupal', 'Magento', 'Shopify', 'E-commerce',
        'Payment Gateways', 'Inventory Management', 'Supply Chain Management', 'Logistics', 'Procurement', 'ERP Systems', 'SAP', 'Oracle',
        'Microsoft Dynamics', 'Tableau', 'Power BI', 'QlikView', 'Looker', 'Data Warehousing', 'ETL', 'Data Engineering', 'Data Governance',
        'Data Quality', 'Master Data Management', 'Predictive Analytics', 'Prescriptive Analytics', 'Descriptive Analytics', 'Business Intelligence',
        'Dashboarding', 'Reporting', 'Data Mining', 'Web Scraping', 'API Integration', 'RESTful APIs', 'GraphQL', 'SOAP', 'Microservices',
        'Serverless Architecture', 'Lambda Functions', 'Event-Driven Architecture', 'Message Queues', 'GraphQL', 'Socket.io', 'WebSockets',
        'Ruby', 'Ruby on Rails', 'PHP', 'Symfony', 'Laravel', 'CakePHP', 'Zend Framework', 'ASP.NET', 'C#', 'VB.NET', 'ASP.NET MVC', 'Entity Framework',
        'Spring', 'Hibernate', 'Struts', 'Kotlin', 'Swift', 'Objective-C', 'iOS Development', 'Android Development', 'Flutter', 'React Native', 'Ionic',
        'Mobile UI/UX Design', 'Material Design', 'SwiftUI', 'RxJava', 'RxSwift', 'Django', 'Flask', 'FastAPI', 'Falcon', 'Tornado', 'WebSockets',
        'GraphQL', 'RESTful Web Services', 'SOAP', 'Microservices Architecture', 'Serverless Computing', 'AWS Lambda', 'Google Cloud Functions',
        'Azure Functions', 'Server Administration', 'System Administration', 'Network Administration', 'Database Administration', 'MySQL', 'PostgreSQL',
        'SQLite', 'Microsoft SQL Server', 'Oracle Database', 'NoSQL', 'MongoDB', 'Cassandra', 'Redis', 'Elasticsearch', 'Firebase', 'Google Analytics',
        'Google Tag Manager', 'Adobe Analytics', 'Marketing Automation', 'Customer Data Platforms', 'Segment', 'Salesforce Marketing Cloud', 'HubSpot CRM',
        'Zapier', 'IFTTT', 'Workflow Automation', 'Robotic Process Automation (RPA)', 'UI Automation', 'Natural Language Generation (NLG)',
        'Virtual Reality (VR)', 'Augmented Reality (AR)', 'Mixed Reality (MR)', 'Unity', 'Unreal Engine', '3D Modeling', 'Animation', 'Motion Graphics',
        'Game Design', 'Game Development', 'Level Design', 'Unity3D', 'Unreal Engine 4', 'Blender', 'Maya', 'Adobe After Effects', 'Adobe Premiere Pro',
        'Final Cut Pro', 'Video Editing', 'Audio Editing', 'Sound Design', 'Music Production', 'Digital Marketing', 'Content Strategy', 'Conversion Rate Optimization (CRO)',
        'A/B Testing', 'Customer Experience (CX)', 'User Experience (UX)', 'User Interface (UI)', 'Persona Development', 'User Journey Mapping', 'Information Architecture (IA)',
        'Wireframing', 'Prototyping', 'Usability Testing', 'Accessibility Compliance', 'Internationalization (I18n)', 'Localization (L10n)', 'Voice User Interface (VUI)',
        'Chatbots', 'Natural Language Understanding (NLU)', 'Speech Synthesis', 'Emotion Detection', 'Sentiment Analysis', 'Image Recognition', 'Object Detection',
        'Facial Recognition', 'Gesture Recognition', 'Document Recognition', 'Fraud Detection', 'Cyber Threat Intelligence', 'Security Information and Event Management (SIEM)',
        'Vulnerability Assessment', 'Incident Response', 'Forensic Analysis', 'Security Operations Center (SOC)', 'Identity and Access Management (IAM)', 'Single Sign-On (SSO)',
        'Multi-Factor Authentication (MFA)', 'Blockchain', 'Cryptocurrency', 'Decentralized Finance (DeFi)', 'Smart Contracts', 'Web3', 'Non-Fungible Tokens (NFTs)'
    ]

    skills = []
    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            skills.append(skill)
    return skills

# Extract education from resume
def extract_education_from_resume(text):
    education_keywords = [
        'Computer Science', 'Information Technology', 'Software Engineering', 'Electrical Engineering', 'Mechanical Engineering', 'Civil Engineering',
        'Chemical Engineering', 'Biomedical Engineering', 'Aerospace Engineering', 'Nuclear Engineering', 'Industrial Engineering', 'Systems Engineering',
        'Environmental Engineering', 'Petroleum Engineering', 'Geological Engineering', 'Marine Engineering', 'Robotics Engineering', 'Biotechnology',
        'Biochemistry', 'Microbiology', 'Genetics', 'Molecular Biology', 'Bioinformatics', 'Neuroscience', 'Biophysics', 'Biostatistics', 'Pharmacology',
        'Physiology', 'Anatomy', 'Pathology', 'Immunology', 'Epidemiology', 'Public Health', 'Health Administration', 'Nursing', 'Medicine', 'Dentistry',
        'Pharmacy', 'Veterinary Medicine', 'Medical Technology', 'Radiography', 'Physical Therapy', 'Occupational Therapy', 'Speech Therapy', 'Nutrition',
        'Sports Science', 'Kinesiology', 'Exercise Physiology', 'Sports Medicine', 'Rehabilitation Science', 'Psychology', 'Counseling', 'Social Work',
        'Sociology', 'Anthropology', 'Criminal Justice', 'Political Science', 'International Relations', 'Economics', 'Finance', 'Accounting', 'Business Administration',
        'Management', 'Marketing', 'Entrepreneurship', 'Hospitality Management', 'Tourism Management', 'Supply Chain Management', 'Logistics Management',
        'Operations Management', 'Human Resource Management', 'Organizational Behavior', 'Project Management', 'Quality Management', 'Risk Management',
        'Strategic Management', 'Public Administration', 'Urban Planning', 'Architecture', 'Interior Design', 'Landscape Architecture', 'Fine Arts',
        'Visual Arts', 'Graphic Design', 'Fashion Design', 'Industrial Design', 'Product Design', 'Animation', 'Film Studies', 'Media Studies',
        'Communication Studies', 'Journalism', 'Broadcasting', 'Creative Writing', 'English Literature', 'Linguistics', 'Translation Studies',
        'Foreign Languages', 'Modern Languages', 'Classical Studies', 'History', 'Archaeology', 'Philosophy', 'Theology', 'Religious Studies',
        'Ethics', 'Education', 'Early Childhood Education', 'Elementary Education', 'Secondary Education', 'Special Education', 'Higher Education',
        'Adult Education', 'Distance Education', 'Online Education', 'Instructional Design', 'Curriculum Development', 'Library Science', 'Information Science',
        'Computer Engineering', 'Software Development', 'Cybersecurity', 'Information Security', 'Network Engineering', 'Data Science', 'Data Analytics',
        'Business Analytics', 'Operations Research', 'Decision Sciences', 'Human-Computer Interaction', 'User Experience Design', 'User Interface Design',
        'Digital Marketing', 'Content Strategy', 'Brand Management', 'Public Relations', 'Corporate Communications', 'Media Production', 'Digital Media',
        'Web Development', 'Mobile App Development', 'Game Development', 'Virtual Reality', 'Augmented Reality', 'Blockchain Technology', 'Cryptocurrency',
        'Digital Forensics', 'Forensic Science', 'Criminalistics', 'Crime Scene Investigation', 'Emergency Management', 'Fire Science', 'Environmental Science',
        'Climate Science', 'Meteorology', 'Geography', 'Geomatics', 'Remote Sensing', 'Geoinformatics', 'Cartography', 'GIS (Geographic Information Systems)',
        'Environmental Management', 'Sustainability Studies', 'Renewable Energy', 'Green Technology', 'Ecology', 'Conservation Biology', 'Wildlife Biology', 'Zoology'
    ]

    education = []
    for keyword in education_keywords:
        pattern = r"(?i)\b{}\b".format(re.escape(keyword))
        match = re.search(pattern, text)
        if match:
            education.append(match.group())
    return education

# Extract name from resume
def extract_name_from_resume(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None

def recommend_courses(test_category, test_score, test_level, top_n=6):
    filtered_df = df[(df['exam_category'] == test_category) &
                     (df['exam_score'] <= test_score) &
                     (df['course_level'] == test_level)]

    if filtered_df.empty:
        return {"message": "No matching courses found"}

    recommended_courses = []
    best_similarity_scores = {}

    for _, row in filtered_df.iterrows():
        course_name = row['course_name']

        if course_name in course_index.index:
            course_idx = course_index[course_name]

            if isinstance(course_idx, pd.Series):
                course_idx = course_idx.iloc[0]

            sim_scores = cosine_sim_mat[course_idx]
            sim_scores[course_idx] = -1

            top_indices = sim_scores.argsort()[-top_n:][::-1]
            for idx in top_indices:
                similar_course = df.iloc[idx]
                score = sim_scores[idx]

                if similar_course['course_id'] not in best_similarity_scores or score > best_similarity_scores[
                    similar_course['course_id']]:
                    best_similarity_scores[similar_course['course_id']] = score
                    recommended_courses.append(similar_course)

    final_recommendations = pd.DataFrame(recommended_courses)
    return final_recommendations['course_id'].unique().tolist()

def summarize_text(text):
    return summarizer(text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]

# Home route
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Resume Parser API. Use the /upload endpoint to upload a resume."})

# Resume upload route
@app.route('/upload', methods=['POST'])
def upload():
    if 'resume' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        if file.filename.endswith('.pdf'):
            text = pdf_to_text(file)
        elif file.filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        else:
            return jsonify({"error": "Invalid file format. Please upload a PDF or TXT file."}), 400

        predicted_category = predict_category(text)
        recommended_job = job_recommendation(text)
        phone = extract_contact_number_from_resume(text)
        email = extract_email_from_resume(text)
        extracted_skills = extract_skills_from_resume(text)
        extracted_education = extract_education_from_resume(text)
        name = extract_name_from_resume(text)

        return jsonify({
            "predicted_category": predicted_category,
            "recommended_job": recommended_job,
            "phone": phone,
            "email": email,
            "name": name,
            "skills": extracted_skills,
            "education": extracted_education
        })
    except PdfReadError:
        return jsonify({"error": "The provided PDF file is corrupted or invalid."}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Course recommendation route
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()

    test_category = data.get('test_category')
    test_score = data.get('test_score')
    test_level = data.get('test_level')

    if not test_category or test_score is None or not test_level:
        return jsonify({"error": "Missing parameters"}), 400

    recommended_courses = recommend_courses(test_category, test_score, test_level)
    return jsonify({"recommended_course_ids": recommended_courses}), 200

# Add course route
@app.route('/add_course', methods=['POST'])
def add_course():
    global df, tfidf_mat, cosine_sim_mat, course_index

    new_course = request.json
    required_fields = ['course_id', 'course_title', 'rating', 'course_level']

    if not all(field in new_course for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    new_course_df = pd.DataFrame([new_course])
    new_course_df['clean_course_name'] = new_course_df['course_title'].apply(nfx.remove_stopwords)
    new_course_df['clean_course_name'] = new_course_df['clean_course_name'].apply(nfx.remove_special_characters)

    new_course_df['combined_features'] = (
        new_course_df['clean_course_name'] + ' ' +
        new_course_df['course_level'].fillna('') + ' ' +
        new_course_df['rating'].astype(str)
    )

    df = pd.concat([df, new_course_df], ignore_index=True)
    tfidf_mat = TfidfVectorizer(stop_words='english').fit_transform(df['combined_features'])
    cosine_sim_mat = cosine_similarity(tfidf_mat)
    course_index = pd.Series(df.index, index=df['course_name']).drop_duplicates()

    return jsonify({"message": "Course added successfully"}), 201

# NLP cleaning route
@app.route('/nlp_clean', methods=['POST'])
def nlp_clean():
    try:
        text = request.json.get('text')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        text = text.translate(str.maketrans('', '', string.punctuation))

        words = nltk.word_tokenize(text.lower())

        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]

        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        cleaned_text = ' '.join(words)

        return jsonify({"cleaned_text": cleaned_text}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Audio transcription route
@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    try:
        data = request.get_json()
        video_url = data.get("video_url")
        print(video_url)
        if not video_url:
            return jsonify({"error": "No video URL provided"}), 400

        response = requests.get(video_url, stream=True)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download video"}), 500

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            temp_video.write(response.content)
            temp_video_path = temp_video.name

        temp_audio_path = temp_video_path.replace(".mp4", ".mp3")
        ffmpeg.input(temp_video_path).output(temp_audio_path, format="mp3").run(overwrite_output=True)

        result = model.transcribe(temp_audio_path)
        transcription = result["text"]

        summary = summarize_text(transcription)

        return jsonify({"transcription": transcription, "summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Flask Error: {e}")
