import pymongo
from api import api
from pymongo.errors import DuplicateKeyError



#MongoDB
uri = "mongodb://localhost:27017/"
client = pymongo.MongoClient(uri)
db = client.TwitterCovidDB
CollectionName = 'Temp'


def loadToDB(tweets):
    '''
        loads data to MongoDB
    '''

    for tweet in tweets:
        try:
            db[CollectionName].collection.insert_one(tweet._json)
        except DuplicateKeyError as e:
            print('Duplicate tweet detected', tweet._json['id'])
            return False



query = "#coronavirus"
tweets = api.search(q=query) #, count=tweets_per_qry
loadToDB(tweets)
print('done')