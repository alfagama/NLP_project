import nltk
from pymongo import MongoClient
nltk.download('twitter_samples')
from nltk.corpus import twitter_samples
from collections import Counter
from data import *
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from random import shuffle
from nltk import classify
from nltk import NaiveBayesClassifier

def bag_of_words(words):
    words_dictionary = dict([word, True] for word in words)
    return words_dictionary


uri = "mongodb://localhost:27017/"
client = MongoClient(uri)
db = client.TwitterCovidDB
CollectionName = 'nltkSamples'
collection = db[CollectionName]

cursor = collection.find({}, {"label":1, "tokens_preprocessed":1, "_id":0})
# positive tweets feature set
pos_tweets_set = []
# negative tweets feature set
neg_tweets_set = []
for document in cursor:
    if document['label'] == 1:
        pos_tweets_set.append((bag_of_words(document['tokens_preprocessed']), 'Positive'))
    else:
        neg_tweets_set.append((bag_of_words(document['tokens_preprocessed']), 'Negative'))


print(pos_tweets_set)  # Output: (5000, 5000)
print(neg_tweets_set)

shuffle(pos_tweets_set)
shuffle(neg_tweets_set)

test_set = pos_tweets_set[:1000] + neg_tweets_set[:1000]
train_set = pos_tweets_set[1000:] + neg_tweets_set[1000:]

print(len(test_set), len(train_set))  # Output: (2000, 8000)

### Naive Bayes Classifier

classifier = NaiveBayesClassifier.train(train_set)
accuracy = classify.accuracy(classifier, test_set)
print(accuracy) # Output: 0.765

## Test in our Tweets
cursor = db['quarantine'].find({}, {"id":1, "full_text":1, "text_preprocessed":1, "tokens_preprocessed":1, "_id":0}).limit(10)
custom_tweet_set = []
for document in cursor:
    print(document['full_text'])
    print(document['text_preprocessed'])
    custom_tweet = document['tokens_preprocessed']
    custom_tweet_set = bag_of_words(custom_tweet)
    print(classifier.classify(custom_tweet_set))
    print()
    label = classifier.classify(custom_tweet_set)
    db['quarantine'].update(
        {
            "id": document["id"]
        },
        {
            "$set": {
                "NBNLTKSamples": label
            }
        }
    )
