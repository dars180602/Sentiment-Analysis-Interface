import streamlit as st
import pickle
import re
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk

# Download NLTK stopwords corpus if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Load the saved model
with open('nb_bin_model3.pkl', 'rb') as file:
    loaded_classifier = pickle.load(file)

# Load the saved vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Define the dictionary of English contractions
contractions_dict = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    # Add more contractions as needed
}

# Define the function to expand contractions
def expand_contractions(text, contractions_dict):
    contractions_pattern = re.compile(r'\b(' + '|'.join(contractions_dict.keys()) + r')\b', flags=re.IGNORECASE)
    processed_dict = {key.lower(): value for key, value in contractions_dict.items()}
    
    def expand_match(contraction):
        match = contraction.group(0)
        expanded_contraction = processed_dict.get(match.lower(), match)
        return expanded_contraction
    
    expanded_text = contractions_pattern.sub(expand_match, text)
    return expanded_text

# Define the preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = expand_contractions(text, contractions_dict)
    tokens = word_tokenize(text)
    
    following_negation = False
    for i in range(len(tokens)):
        token = tokens[i]
        if token in ["not", "no"]:
            following_negation = True
        elif following_negation:
            tokens[i] = "not_" + tokens[i]
            following_negation = False
    
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [token for token in tokens if token.isalnum() or token == '_']
    tokens = [token.strip() for token in tokens if token is not None]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Create the Streamlit UI
st.title('Sentiment Analysis App')
st.write('Enter your review below:')

# Create a text input for user to input review
review_input = st.text_area('Input your review here:', '')

# Make prediction when button is clicked
if st.button('Predict Sentiment'):
    if review_input:
        # Preprocess the input review
        preprocessed_review = preprocess_text(review_input)
        
        # Transform preprocessed review data using the TF-IDF vectorizer
        review_tfidf = tfidf_vectorizer.transform([preprocessed_review])

        # Make predictions using the loaded model
        prediction = loaded_classifier.predict(review_tfidf)

        # Convert prediction to sentiment label
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        
        # Display the sentiment prediction
        st.write(f'The sentiment of the review is: {sentiment}')
    else:
        st.write('Please enter a review before predicting.')
