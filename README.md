# Sentiment Analysis on Yelp Dataset Reviews
## Overview
This project aims to perform sentiment analysis on the Yelp dataset reviews. Sentiment analysis is the process of determining the sentiment expressed in a piece of text, which can be positive, negative, or neutral. By analyzing Yelp reviews, we can gain insights into the sentiments of customers towards various businesses, helping businesses to understand their strengths and weaknesses.

## Dataset
The dataset used in this project is the Yelp dataset, which contains millions of reviews, business attributes, and geographical data for businesses across different cities. The dataset is publicly available and can be downloaded from the official Yelp dataset challenge website: Yelp Dataset Challenge

## Tools and Libraries
Python: Programming language used for data manipulation, analysis, and visualization.
Jupyter Notebook: Interactive development environment used for data exploration and analysis.
Pandas: Python library for data manipulation and analysis.
NLTK (Natural Language Toolkit): Library for natural language processing tasks such as tokenization, stemming, and sentiment analysis.
Scikit-learn: Library for machine learning tasks including text feature extraction and classification.
Matplotlib and Seaborn: Python libraries for data visualization.

## Methodology
Data Preprocessing: Cleaning and preprocessing the Yelp reviews dataset, which includes removing punctuation, stop words, and performing tokenization.

Feature Extraction: Extracting features from the preprocessed text data using techniques like TF-IDF (Term Frequency-Inverse Document Frequency).

Model Training: Training a machine learning model (e.g., Support Vector Machine, Naive Bayes) using the extracted features to classify reviews into positive, negative, or neutral sentiments.

Evaluation: Evaluating the performance of the trained model using metrics such as accuracy, precision, recall, and F1-score.

Deployment: Deploying the trained model to classify sentiments of new reviews.

## Results

Accuracy

BNB Model   88.2% (Test) - 83.3% (Training)

LSTM Model  95.2% (Test) - 93.8% (Training)


## Usage
Clone the repository:
bash
Copy code
git clone https://github.com/your-username/sentiment-analysis-yelp.git
Install the required dependencies:
Copy code
pip install -r requirements.txt
Run the Jupyter Notebook sentiment_analysis_yelp.ipynb to explore the data, preprocess it, train the model, and evaluate its performance.
Deploy the trained model for sentiment analysis on new Yelp reviews.

## Contributors
Abhikumar Patel
Cecille Jatulan
David Higuera
Diana Reyes
Maria Melencio
Michael Montanez

