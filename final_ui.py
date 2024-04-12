import streamlit as st
import pickle
import re
import unicodedata
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


# Download NLTK stopwords corpus if not already downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Load the model
loaded_classifier = load_model('lstm_model1.h5')

# Load the saved vectorizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Dictionary of English contractions
contractions_dict = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
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

# Add an image to your Streamlit app
image = st.image('sentiment.webp')

st.write("""
## About
**Welcome to the Sentiment Analysis App! This app allows you to analyze the sentiment whether a Review is Positive or Negative.**

The notebook, model and documentation are available on [GitHub.](https://github.com/dars180602/Sentiment-Analysis-Interface)        

**Contributors:** 
- **Cecille Jatulan**
- **David Higuera**
- **Diana Reyes**
- **Mike Montanez**
- **Maria Melencio**
- **Abhikumar Patel**
         
""")

st.write('Enter your review below:')

# Create a text input for user to input review
review_input = st.text_area('Input your review here:', '')

# Make prediction when button is clicked
if st.button('Predict Sentiment'):
    if review_input:
        # Preprocess the input review
        preprocessed_review = preprocess_text(review_input)
        
        # Tokenize and pad the text instance
        text_seq = tokenizer.texts_to_sequences([preprocessed_review])
        text_pad = pad_sequences(text_seq, maxlen=100)

        # Make predictions using the loaded model
        prediction = loaded_classifier.predict(text_pad)

        # Display the numerical predicstion value
        st.write(f'Numerical prediction value: {prediction}')

        # Convert prediction to sentiment label
        if prediction > 0.5:
            sentiment = 'Positive'  
        else:
            sentiment='Negative'
        
        # Display the sentiment prediction
        st.write(f'The sentiment of the review is: {sentiment}')
    else:
        st.write('Please enter a review before predicting.')
