import nltk
from pymongo import MongoClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

#####################################################################
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

#####################################################################
connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterDB
collections = db.collection_names()


#####################################################################
def update_labels(compount_pre, compount_full, polarity_pre, polarity_full):
    db[collection].update(
        {
            "id": tweet["id"]
        },
        {
            "$set": {
                "vader_preprocessed_label": compount_pre,
                "vader_full_text_label": compount_full,
                "textblob_preprocessed_label": polarity_pre,
                "textblob_full_text_label": polarity_full
            }
        }
    )


#####################################################################
counter = 0
for collection in collections:
    tweets = db[collection].find().batch_size(10)
    if collection == 'vaccine_test3':
        for tweet in tweets:
            #   get full_text and text_preprocessed
            text_pre = tweet["text_preprocessed"]
            text_full = tweet["full_text"]
            #   vader preprocessed
            v_scores_pre = sid.polarity_scores(text_pre)
            v_label_pre = 1 if v_scores_pre['compound'] > 0 else -1 if v_scores_pre['compound'] < 0 else 0
            #   textblob preprocessed
            t_polarity_preprocessed = TextBlob(text_pre).sentiment.polarity
            t_label_pre = 1 if t_polarity_preprocessed > 0 else -1 if t_polarity_preprocessed < 0 else 0
            #   vader full text
            v_scores_full = sid.polarity_scores(text_full)
            v_label_full = 1 if v_scores_full['compound'] > 0 else -1 if v_scores_full['compound'] < 0 else 0
            #   textblob full text
            t_polarity_full = TextBlob(text_full).sentiment.polarity
            t_label_full = 1 if t_polarity_full > 0 else -1 if t_polarity_full < 0 else 0
            #   call method and add labels
            update_labels(v_label_pre, v_label_full, t_label_pre, t_label_full)
