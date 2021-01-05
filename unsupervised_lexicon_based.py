import nltk
from pymongo import MongoClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

#   ##################################################################################################
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

#   ##################################################################################################
connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterDB
collections = db.collection_names()


#   ##################################################################################################
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


#   ##################################################################################################
counter = 0
vader_pre_neg, vader_full_neg, blob_pre_neg, blob_full_neg = 0, 0, 0, 0
vader_pre_pos, vader_full_pos, blob_pre_pos, blob_full_pos = 0, 0, 0, 0
vader_pre_neutral, vader_full_neutral, blob_pre_neutral, blob_full_neutral = 0, 0, 0, 0
same_pos, same_neg, same_neutral = 0, 0, 0
same_pos_full, same_neg_full, same_neutral_full = 0, 0, 0
#   ##################################################################################################
for collection in collections:
    tweets = db[collection].find().batch_size(10)
    if collection == 'vaccine_preprocessed':
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
#   ##################################################################################################
            #   print tweet counter
            counter += 1
            print(counter)
            #   if vader preprocessed
            if v_label_pre == 1:
                vader_pre_pos += 1
            elif v_label_pre == 0:
                vader_pre_neutral += 1
            else:
                vader_pre_neg += 1
            #   if vader full
            if v_label_full == 1:
                vader_full_pos += 1
            elif v_label_full == 0:
                vader_full_neutral += 1
            else:
                vader_full_neg += 1
            #   if textblob preprocessed
            if t_label_pre == 1:
                blob_pre_pos += 1
            elif t_label_pre == 0:
                blob_pre_neutral += 1
            else:
                blob_pre_neg += 1
            #   if textblob full
            if t_label_full == 1:
                blob_full_pos += 1
            elif t_label_full == 0:
                blob_full_neutral += 1
            else:
                blob_full_neg += 1
            #   same values pre
            if v_label_pre == t_label_pre:
                if v_label_pre == 1:
                    same_pos += 1
                elif v_label_pre == 0:
                    same_neutral += 1
                else:
                    same_neg += 1
            #   same values full
            if v_label_full == t_label_pre:
                if v_label_full == 1:
                    same_pos_full += 1
                elif v_label_full == 0:
                    same_neutral_full += 1
                else:
                    same_neg_full += 1
#   print all
print("Vader pre: pos", vader_pre_pos, ", neutral:", vader_pre_neutral, ", neg:", vader_pre_neg)
print("Vader full: pos", vader_full_pos, ", neutral:", vader_full_neutral, ", neg:", vader_full_neg)
print("Blob pre: pos", blob_pre_pos, ", neutral:", blob_pre_neutral, ", neg:", blob_pre_neg)
print("Blob full: pos", blob_full_pos, ", neutral:", blob_full_neutral, ", neg:", blob_full_neg)
print("Same values in: pos", same_pos, ", neutral:", same_neutral, ", neg:", same_neg)
print("Same values in: pos", same_pos_full, ", neutral:", same_neutral_full, ", neg:", same_neg_full)
