from FileAlexia.api import api
import tweepy
import time
from mongodb_useful.mongo_db import *

query = "#coronavirus OR #covid OR #covid19 OR #covid_19"
CollectionName = 'covid2'
collection = db[CollectionName]


def loadToDB(tweet):
    '''
        loads data to MongoDB
    '''

    collection.insert_one(tweet._json)
    print('[Insert]', tweet._json['id'], 'for query', query, '[' + str(tweet._json['created_at']) + ']','[' + str(collection.estimated_document_count()) + ']')

    '''
    #limit: provides a way to find if there is at least one matching occurrence. Limiting the number of matching occurrences makes the collection scan stop as soon as a match is found instead of ' \
    #going through the whole collection.
    if collection.count_documents({'id': tweet._json['id'] }, limit = 1) != 0:
        print('Tweet with id ', tweet._json['id'], ' already exists')
    else:
        collection.insert_one(tweet._json)
        print('[Insert]', tweet._json['id'], 'for query', query,  '[' + str(tweet._json['created_at']) + ']', '[' + str(collection.estimated_document_count()) + ']')
    '''

#search with tweepy for tweets in english language only and excluding retweets
tweets = tweepy.Cursor(api.search,
                       q=query + " -filter:retweets",
                       lang="en", tweet_mode='extended').items()


while True:
    try:
        tweet = tweets.next()
        loadToDB(tweet)
    except tweepy.TweepError:
        print("sleeping")
        time.sleep(60 * 15)
        continue
    except StopIteration:
        break




