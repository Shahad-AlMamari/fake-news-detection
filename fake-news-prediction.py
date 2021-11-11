import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.text import Text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import streamlit as st
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

    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    return corpus

# Loading the Models    
classifier_model = joblib.load('RF_classifier.pkl')
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,3),vocabulary=joblib.load(open("tfidf_features.pkl","rb")))

# Generating and Displaying Predictions
def classify_news(fmodel,cmodel, news):

    #tfidf     
    #tfidf = vectorizer.fit_transform(preprocessor(news))
    #rf classifiation
    #label = classifier_model.predict(tfidf)[0]
    tfidf = fmodel.fit_transform(preprocessor(news))
    #rf classifiation
    label = cmodel.predict(tfidf)[0]
    if label == 1:
        prediction = 'Real'
    elif label == 0:
        prediction = 'Fake' 
    pred = "label", label    
    return {'label': label}
    #return pred
    
# output the model’s predictions as a dictionary
if news_text != '':
    #result = preprocessor(news_text)    
    result = classify_news(vectorizer, classifier_model, news_text)

    st.write(result)