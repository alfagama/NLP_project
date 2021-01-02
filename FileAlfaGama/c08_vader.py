import nltk
from pymongo import MongoClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#####################################################################
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

#####################################################################
connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterDB
collections = db.collection_names()


#####################################################################
def update_label_with_vader(compount_pre, compount_full):
    db[collection].update(
        {
            "id": tweet["id"]
        },
        {
            "$set": {
                "vader_preprocessed_label": compount_pre,
                "vader_full_text_label": compount_full
            }
        }
    )


#####################################################################
counter = 0
for collection in collections:
    tweets = db[collection].find().batch_size(10)
    if collection == 'vaccine_test3':
        for tweet in tweets:
            text_pre = tweet["text_preprocessed"]
            scores_pre = sid.polarity_scores(text_pre)
            label_pre = 1 if scores_pre['compound'] > 0 else -1 if scores_pre['compound'] < 0 else 0
            #print(scores_pre)
            text_full = tweet["full_text"]
            scores_full = sid.polarity_scores(text_full)
            label_full = 1 if scores_full['compound'] > 0 else -1 if scores_full['compound'] < 0 else 0
            #print(scores_full)
            update_label_with_vader(label_pre, label_full)
            counter += 1
            print(counter)
