import pymongo
from api import api
from pymongo.errors import DuplicateKeyError
import tweepy
import time

query = "#coronavirus"

#MongoDB
uri = "mongodb://localhost:27017/"
client = pymongo.MongoClient(uri)
db = client.TwitterCovidDB
CollectionName = 'temp'
collection = db[CollectionName]


def loadToDB(tweet):
    '''
        loads data to MongoDB
    '''

    try:

        collection.insert_one(tweet._json)
        print('[Insert]', tweet._json['id'], 'for query', query, '[' + str(collection.estimated_document_count()) + ']')

    except DuplicateKeyError as e:
        print('Duplicate tweet detected', tweet._json['id'])
        #return False


tweets = tweepy.Cursor(api.search,
              q=query,
              lang="en").items()

i = 0
while True:
    try:
        tweet = tweets.next()
        print(i)
        i += 1
        #loadToDB(tweets)
    except tweepy.TweepError:
        time.sleep(60 * 15)
        continue
    except StopIteration:
        break


'''i =0
for tweet in tweets.items():
    print(i)
    i+=1
    #loadToDB(tweets)'''




