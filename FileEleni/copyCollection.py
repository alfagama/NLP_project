from pymongo import MongoClient
import timeit

#####################################################################
connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterCovidDB

collection = db['quarantine']

start = timeit.default_timer()

######################################
collectionName = 'quarantine'

db[collectionName].aggregate([ { "$match": {} }, { "$out": "copiedCollection" } ])

######################################
stop = timeit.default_timer()
print('Time: %.5f' % (stop - start))
######################################
