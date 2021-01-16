from pymongo import MongoClient
import nltk
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('wordnet')
nltk.download('vader_lexicon') # do this once: grab the trained model from the web


# function to print sentiments
# of the sentence.
def sentiment_scores(sentence):

    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    # polarity_scores method of SentimentIntensityAnalyzer
    # oject gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)
    """
    print("Overall sentiment dictionary is : ", sentiment_dict)
    print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
    print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
    print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")

    print("Sentence Overall Rated As", end = " ")
    """
    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05:
        print("Positive")
        finalLabel = "Positive"
    elif sentiment_dict['compound'] <= - 0.05:
        print("Negative")
        finalLabel = "Negative"
    else :
        print("Neutral")
        finalLabel = "Neutral"
    print()
    return finalLabel

def getLabel(score):
    if score > 0.0:
        finalLabel = "Positive"
    elif score < 0.0:
        finalLabel = "Negative"
    else:
        finalLabel = "Neutral"
    print(finalLabel)
    return finalLabel

connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterCovidDB
collectionName = 'Sent140'
collection = db[collectionName]

samePositives = 0
sameNegatives = 0
sameNeutrals = 0
differences = 0
fullDif = 0
cursor = collection.find(no_cursor_timeout=True).batch_size(20)
#cursor = db['quarantine'].find({"text_preprocessed": {"$exists": True}}, {"full_text":1, "text_preprocessed":1, "tokens_preprocessed":1, "_id":0}).limit(10)
print(cursor.count())
errorIds = []
i = 0
for document in cursor:

    tweet = document["text_preprocessed"]
    labelVader = sentiment_scores(tweet)
    labelTextBlob = getLabel(TextBlob(tweet).sentiment.polarity)
    print(labelVader, ' ', labelTextBlob)

    db[collectionName].update(
        {
            "_id": document["_id"]
        },
        {
            "$set": {
                "Vader": labelVader,
                "TextBlob": labelTextBlob
            }
        }
    )
    if labelVader == labelTextBlob:
        if labelVader == 'Positive':
            samePositives += 1
        elif labelVader == 'Neutral':
            sameNeutrals += 1
        else:
            sameNegatives += 1
    else:
        differences += 1
        if labelVader != 'Neutral' and labelTextBlob != 'Neutral':
            fullDif +=1


    i = i+1
    print("counter :", i)


print("Same positives", samePositives)
print("Same negatives", sameNegatives)
print("Same neutrals", sameNeutrals)
print("Differences", differences)
print("Full diff", fullDif)