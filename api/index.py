from flask import Flask, request, jsonify
import pandas as pd
import neattext.functions as nfx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import ffmpeg
import whisper
import tempfile
from transformers import pipeline

app = Flask(__name__)
CORS(app,supports_credentials=True)
# Load dataset
DATASET_PATH = "C:/Users/karth/Downloads/user_course_data_test.csv"
df = pd.read_csv(DATASET_PATH)

df['clean_course_name'] = df['course_name'].apply(nfx.remove_stopwords)
df['clean_course_name'] = df['clean_course_name'].apply(nfx.remove_special_characters)

# Combine Features
df['combined_features'] = (
    df['clean_course_name'] + ' ' +
    df['course_level'].fillna('') + ' ' +
    df['rating'].astype(str) + ' ' +
    df['exam_category'].fillna('') + ' ' +
    df['exam_score'].astype(str)
)

# Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_mat = tfidf.fit_transform(df['combined_features'])

# Cosine Similarity
cosine_sim_mat = cosine_similarity(tfidf_mat)

# Course Index Mapping
course_index = pd.Series(df.index, index=df['course_name']).drop_duplicates()
CORS(app)

model = whisper.load_model("base")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

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

@app.route("/transcribe", methods=["POST","OPTIONS"])
def transcribe_audio():
    if request.method == "OPTIONS":  # Handle CORS preflight request
        response = jsonify({"message": "CORS preflight successful"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        return response, 204  # No Content
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

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Flask Error: {e}")
