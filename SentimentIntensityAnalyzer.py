from pymongo import MongoClient
from collections import Counter
import nltk
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pymongo.errors import DuplicateKeyError
from data import *
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('vader_lexicon') # do this once: grab the trained model from the web
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer


# function to print sentiments
# of the sentence.
def sentiment_scores(sentence):

    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    # polarity_scores method of SentimentIntensityAnalyzer
    # oject gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)

    print("Overall sentiment dictionary is : ", sentiment_dict)
    print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
    print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
    print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")

    print("Sentence Overall Rated As", end = " ")

    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05 :
        print("Positive")
        finalLabel = "Positive"
    elif sentiment_dict['compound'] <= - 0.05 :
        print("Negative")
        finalLabel = "Negative"
    else :
        print("Neutral")
        finalLabel = "Neutral"
    print()
    return finalLabel



connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterCovidDB
#collection = db['quarantine']

#db['quarantine'].create_index( [( "tokens" , 1 )] )
cursor = db['quarantine'].find({}, {"full_text":1, "text_preprocessed":1, "tokens_preprocessed":1, "_id":0, "id":1}).limit(500000)
#cursor = db['quarantine'].find({"text_preprocessed": {"$exists": True}}, {"full_text":1, "text_preprocessed":1, "tokens_preprocessed":1, "_id":0}).limit(10)
print(cursor.count())
errorIds = []
i = 0
for document in cursor:
    try:
        tweet = document["text_preprocessed"]
        label = sentiment_scores(tweet)
        db['quarantine'].update(
            {
                "id": document["id"]
            },
            {
                "$set": {
                    "SentimentIntensityAnalyzer": label
                }
            }
        )
    except (KeyError):
        #print("I have a keyError in ", document["id"])
        errorIds.append(document["id"])
        pass
    i = i +1
    print(i)

print("I have a keyError in ", errorIds)
print(len(errorIds))
