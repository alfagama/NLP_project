from pymongo import MongoClient
from textblob import TextBlob

#####################################################################
connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterDB
collections = db.collection_names()


#####################################################################
def update_label_with_textblob(polarity_pre, polarityt_full):
    db[collection].update(
        {
            "id": tweet["id"]
        },
        {
            "$set": {
                "textblob_preprocessed_label": polarity_pre,
                "textblob_full_text_label": polarityt_full
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
            polarity_preprocessed = TextBlob(text_pre).sentiment.polarity
            label_pre = 1 if polarity_preprocessed > 0 else -1 if polarity_preprocessed < 0 else 0
            print(polarity_preprocessed)
            text_full = tweet["full_text"]
            polarity_full = TextBlob(text_full).sentiment.polarity
            label_full = 1 if polarity_full > 0 else -1 if polarity_full < 0 else 0
            print(polarity_full)
            update_label_with_textblob(label_pre, label_full)
            counter += 1
            print(counter)
