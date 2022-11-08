import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import string
from matplotlib import style
style.use('ggplot')
import tweepy
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')
from nltk.corpus import wordnet

import streamlit as st


st.title("Twitter sentiment analysis")
keyword=st.text_input("",placeholder="Please type your tweet here")

if st.button("Analysis"):
    consumer_key = "LFWaB0IiFY3ubTWZEbRpCESy4"
    consumer_sec = "GMCJwqgphQfUm7KdaYBuxPDfgmMcqk9QibsMNIFW9J9EXGdrkH"
    access_token = "1562360537496711168-utZWhPYz0i5ujGfNVwm9J0yLQwRLOT"
    access_token_sec = "zBXjUZ8Hf2NrwLL1URZXmk3umVB3BeBrT1idVv8RnAp2x"
    auth = tweepy.OAuthHandler(consumer_key, consumer_sec)
    auth.set_access_token(access_token, access_token_sec)
    api = tweepy.API(auth)
    #keyword = input("Please enter keyword or hashtag to search: ")
    noOfTweet = 100
    # tweet_data=tweepy.Cursor(api.search_tweets, q=keyword).items(noOfTweet)
    tweet_data = api.search_tweets(q=keyword, count=200)
    df = []
    for tweet in tweet_data:
        df.append(tweet.text)
    df = pd.DataFrame(df)

    df.columns = ["tweet"]
    text_df = df


    def data_processing(text):
        alphaPattern = "[^a-zA-Z0-9]"
        sequencePattern = r"(.)\1\1+"
        seqReplacePattern = r"\1\1"
        text = text.lower()
        text = re.sub(r"https\S+|www\S+https\S+", '', text, flags=re.MULTILINE)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r'\@w+|\#', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(alphaPattern, " ", text)
        text = re.sub(sequencePattern, seqReplacePattern, text)
        text_tokens = word_tokenize(text)
        filtered_text = [w for w in text_tokens if not w in stop_words]

        # ps=PorterStemmer()
        # stemmed_words=[ps.stem(w) for w in filtered_text]

        lm = WordNetLemmatizer()
        lemmatized = [lm.lemmatize(word, wordnet.VERB) for word in filtered_text]

        return " ".join(lemmatized)


    text_df['tweet'] = text_df['tweet'].apply(data_processing)


    def polarity(text):
        return TextBlob(text).sentiment.polarity


    text_df['polarity'] = text_df['tweet'].apply(polarity)


    def sentiment(label):
        if label < 0:
            return "Negative"
        elif label == 0:
            return "Neutral"
        elif label > 0:
            return "Positive"


    text_df['sentiment'] = text_df['polarity'].apply(sentiment)

    fig = plt.figure(figsize=(4, 4))
    colors = ("yellowgreen", "cyan", "red")
    wp = {'linewidth': 2, 'edgecolor': "black"}
    tags = text_df['sentiment'].value_counts()
    explode = (0.1, 0.1, 0.1)
    tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors=colors,
              startangle=90, wedgeprops=wp, explode=explode, label='')
    #plt.title('Distribution of sentiments')
    plt.savefig("text.png", bboxinches="tight")
    img = Image.open('text.png')
    st.image(img, width=400, caption="Twitter Sentiment Analysis")