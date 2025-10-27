import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Configuration ---
MODEL_FILE = 'rf_baseline_model.joblib'
VECTORIZER_FILE = 'tfidf_vectorizer.joblib'

# Initialize NLTK components (if you use them for text processing)
try:
    STOPWORDS = set(stopwords.words('english'))
    LEMMA = WordNetLemmatizer()
    
except LookupError:
    # Handle case where NLTK data might not be downloaded in the deployment environment
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    STOPWORDS = set(stopwords.words('english'))
    LEMMA = WordNetLemmatizer()


# --- Text Preprocessing Function (MUST MATCH Training Preprocessing) ---
def preprocess_text(text):
    """Cleans and processes the input text."""
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs, Emojis, and HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+|[^\w\s@#\.\,-]+', '', text, flags=re.MULTILINE)
    
    # 3. Remove punctuation and non-word characters (keeping space)
    text = re.sub(r'[^a-z\s]', '', text) 
    
    # 4. Tokenization, Stopword Removal, and Lemmatization
    tokens = text.split()
    processed_tokens = []
    for token in tokens:
        if token not in STOPWORDS and len(token) > 1:
            processed_tokens.append(LEMMA.lemmatize(token))
            
    return " ".join(processed_tokens)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded_string});
            background-size: cover;
            background-attachment: fixed; /* Optional: Makes the background scroll with content */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Call the function at the start of your script ---
# You must import 'base64' for this to work
import base64
add_bg_from_local('bg.jpg')

# --- Load Model and Vectorizer (Use Caching for Performance) ---
# @st.cache_resource is used to load complex objects like models only once
@st.cache_resource
def load_assets():
    """Loads the model and vectorizer from joblib files."""
    try:
        # Load the saved model and vectorizer
        vectorizer = joblib.load(VECTORIZER_FILE)
        model = joblib.load(MODEL_FILE)
        return vectorizer, model
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e.filename}. Make sure '{MODEL_FILE}' and '{VECTORIZER_FILE}' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        st.stop()


# Load the assets
tfidf_vectorizer, rf_model = load_assets()


# --- Streamlit Application Layout and Logic ---
st.title("DeepSaturn: CSAT Prediction from Customer Feedback 💬")
st.markdown("---")


# Text Area for Input
feedback_text = st.text_area(
    "Enter Customer Feedback Text:",
    "The product arrived late and was damaged, completely ruining the experience.",
    height=150
)

# Button to trigger prediction
if st.button("Predict CSAT Score"):
    
    # 1. Preprocess the text
    processed_text = preprocess_text(feedback_text)
    
    if not processed_text:
        st.warning("Please enter some meaningful text for prediction.")
    else:
        # 2. Vectorize the text
        # The vectorizer MUST transform the text into the exact feature space (5028 features)
        X_sparse_input = tfidf_vectorizer.transform([processed_text])
        
        # 3. Predict the CSAT score (1 to 5)
        # Model returns the predicted class (1, 2, 3, 4, or 5)
        predicted_csat = rf_model.predict(X_sparse_input)[0]
        
        # 4. Get probability/confidence
        # Model returns an array of probabilities for each class [P(1), P(2), P(3), P(4), P(5)]
        prediction_proba = rf_model.predict_proba(X_sparse_input)[0]
        
        # Get the confidence for the predicted class
        confidence_index = predicted_csat - 1  # CSAT 1 is index 0, CSAT 5 is index 4
        confidence = prediction_proba[confidence_index] * 100
        
        # --- Display Results ---
        
        st.subheader("Prediction Result")
        
        if predicted_csat <= 2:
            st.error(f"Predicted CSAT: {predicted_csat} (Detractor/Dissatisfied)")
        elif predicted_csat == 3:
            st.warning(f"Predicted CSAT: {predicted_csat} (Neutral/Passive)")
        else:
            st.success(f"Predicted CSAT: {predicted_csat} (Promoter/Satisfied)")

        st.info(f"Confidence Level: **{confidence:.2f}%**")
        
        st.markdown("---")
        st.subheader("Probability Distribution")
        
        # Display the distribution of probabilities
        classes = [1, 2, 3, 4, 5]
        proba_df = st.dataframe({
            'CSAT Score': classes,
            'Probability (%)': [f'{p * 100:.2f}' for p in prediction_proba]
        }, hide_index=True)
        
        st.caption("Note: The text processing function in this app must exactly match the one used during model training.")