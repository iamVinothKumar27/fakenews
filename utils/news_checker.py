import os
import nltk
import string
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import utils.dependencies as deps

# Load environment variables from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize data storage for scraped news
news_data = {}
similarity_threshold = 0.7   # how strict tf-idf match should be

# Configure Gemini API for Gemini calls (if key is available)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY not set; Gemini verification will be skipped.")

def preprocess_text(text):
    """Tokenize, remove stopwords and pad using the shared tokenizer/model deps."""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower(), preserve_line=True)
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

    if deps.TOKENIZER is None:
        raise RuntimeError("TOKENIZER is not initialized. Make sure load_dependencies() was called before predictions.")

    sequence = deps.TOKENIZER.texts_to_sequences([" ".join(filtered_tokens)])
    return pad_sequences(sequence, maxlen=deps.max_length)

def predict_news(text):
    """Predict whether news is Fake/Real using the trained Keras model."""
    if deps.model is None:
        raise RuntimeError("Model is not initialized. Make sure load_dependencies() was called before predictions.")

    processed_text = preprocess_text(text)
    prediction = deps.model.predict(processed_text)[0][0]
    return "Fake" if prediction >= 0.5 else "Real"

def update_news_data(new_data):
    """Merge newly scraped news into the global news_data dict."""
    global news_data
    for site, df in new_data.items():
        if site in news_data:
            news_data[site] = news_data[site].append(df, ignore_index=True)
        else:
            news_data[site] = df

def find_similar_news(input_text):
    """Return the most similar news article content using TF-IDF cosine similarity."""
    global news_data
    all_news = []
    for site, df in news_data.items():
        all_news.extend(df["Content"].values)

    if not all_news:
        return None

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_news + [input_text])

    # Last row is the input text; compare it to all previous rows (the crawled news)
    similarities = cosine_similarity(
        tfidf_matrix[-1],   # input text vector
        tfidf_matrix[:-1]   # all scraped news vectors
    ).flatten()

    if similarities.max() >= similarity_threshold:
        return all_news[similarities.argmax()]
    return None

def verify_with_gemini(news_input, matched_news):
    """Use Gemini to check if two headlines mean the same thing. Returns 'Yes' or 'No'."""
    if not GEMINI_API_KEY:
        # If no API key is configured, fall back to assuming they match
        return "Yes"

    prompt = f"""Compare the following two news headlines and tell me if they mean the same thing.
Reply with only 'Yes' or 'No'.

1. {news_input}
2. {matched_news}
"""
    response = genai.GenerativeModel('gemini-2.0-flash').generate_content(prompt)
    return (response.text or "Yes").strip()
