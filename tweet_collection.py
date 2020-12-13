import pymongo
from api import api
from pymongo.errors import DuplicateKeyError
import tweepy
import time

query = "#coronavirus OR #covid OR #covid19"

#MongoDB
uri = "mongodb://localhost:27017/"
client = pymongo.MongoClient(uri)
db = client.TwitterCovidDB
CollectionName = 'covid2'
collection = db[CollectionName]


def loadToDB(tweet):
    '''
        loads data to MongoDB
    '''

    try:

        collection.insert_one(tweet._json)
        print('[Insert]', tweet._json['id'], 'for query', query, '[' + str(collection.estimated_document_count()) + ']')
        print(tweet._json['full_text'])

    except DuplicateKeyError as e:
        print('Duplicate tweet detected', tweet._json['id'])
        #return False


tweets = tweepy.Cursor(api.search,
              q=query + " -filter:retweets",
              lang="en", tweet_mode='extended').items(50)

#i = 0
while True:
    try:
        tweet = tweets.next()
        #print(i)
        #i += 1
        loadToDB(tweet)
    except tweepy.TweepError:
        time.sleep(60 * 15)
        continue
    except StopIteration:
        break







