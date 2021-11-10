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

st.set_page_config(page_title="News Detection", page_icon="news-logo.png")


# Creating a Header
st.write("# Fake News Detector\n Detect fake news by either inserting the news article text")

# Adding Text Input
news_text = st.text_area("Enter text for prediction")

# pre-processing
def preprocessor(text):
    wordnet=WordNetLemmatizer()
    corpus = []

    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()

    text = [wordnet.lemmatize(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    corpus.append(text)
    return corpus

# Loading the Models    
loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("features_tfidf.pkl", "rb")))
vlassifier_model = joblib.load('RandomForest_model.pkl')

# Generating and Displaying Predictions
def classify_news(fmodel,cmodel, news):

    #tfidf 
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(loaded_vec.fit_transform(np.array(list(preprocessor(news)))))
    #rf classifiation
    label = vlassifier_model.predict(tfidf)[0]
    return {'label': label}
'''    if label == 1:
        prediction = 'Real'
    elif label == 0:
        prediction = 'Fake' 
        
    return {'label': prediction}'''
    
# output the model’s predictions as a dictionary
if news_text != '':
    #result = preprocessor(news_text)    
    result = classify_news(loaded_vec,vlassifier_model, news_text)

    st.write(result)

    #result = vectorize_news(loaded_vec,news_text)

