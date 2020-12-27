import sys
import pymongo
from pymongo import MongoClient

#####################################################################
connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterDB

uri = "mongodb://localhost:27017/"
client = pymongo.MongoClient(uri)
CollectionName = 'vaccine_test3'
collection_test = db[CollectionName]

#####################################################################


def load_to_db(tweet_to_load):
    collection_test.insert_one(tweet_to_load)


#####################################################################
collections = db.collection_names()

#####################################################################
for collection in collections:
    if collection == 'vaccine':
        tweets = db[collection].find()
        for tweet in tweets:
            load_to_db(tweet)
        break
    else:
        continue
#####################################################################
