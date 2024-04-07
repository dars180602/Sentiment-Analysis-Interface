import streamlit as st
import pickle

# Load the saved model
with open('nb_bin_model3.pkl', 'rb') as file:
    loaded_classifier = pickle.load(file)

# Load the saved vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Define the function to make predictions
def predict_sentiment(review):
    # Preprocess test data if necessary (ensure it's in the same format as during training)

    # Transform test data using the TF-IDF vectorizer
    review_tfidf = tfidf_vectorizer.transform([review])

    # Make predictions using the loaded model
    prediction = loaded_classifier.predict(review_tfidf)

    return prediction

# Create the Streamlit UI
st.title('Sentiment Analysis App')
st.write('Enter your review below:')

# Create a text input for user to input review
review_input = st.text_area('Input your review here:', '')

# Make prediction when button is clicked
if st.button('Predict Sentiment'):
    if review_input:
        prediction = predict_sentiment(review_input)
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        st.write(f'The sentiment of the review is: {sentiment}')
    else:
        st.write('Please enter a review before predicting.')
