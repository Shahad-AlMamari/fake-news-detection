import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.text import Text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import joblib
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
import numpy as np
from nltk.stem.porter import PorterStemmer


st.set_page_config(page_title="News Detection", page_icon="news-logo.png")


# Creating a Header
st.write("# Fake News Detector\n Detect fake news by either inserting the news article text")

# Adding Text Input
news_text = st.text_area("Enter text for prediction")

# pre-processing
def preprocessor(text):
    ps = PorterStemmer()
    corpus = []

    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()

    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    corpus.append(text)
    return corpus

# Loading the Models    
vlassifier_model = joblib.load('RF_model.pkl')

# Generating and Displaying Predictions
def classify_news(cmodel, news):

    #tfidf 
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,3),vocabulary=joblib.load(open("tfidf_features.pkl","rb")))
    tfidf = vectorizer.fit_transform(preprocessor(news))
    #rf classifiation
    label = vlassifier_model.predict(tfidf)[0]
    if label == 1:
        prediction = 'Real'
    elif label == 0:
        prediction = 'Fake' 
        
    return {'label': prediction}
    
# output the model’s predictions as a dictionary
if news_text != '':
    #result = preprocessor(news_text)    
    result = classify_news(vlassifier_model, news_text)

    st.write(result)

