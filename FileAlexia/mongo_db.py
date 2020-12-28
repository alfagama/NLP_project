import pymongo

#MongoDB
uri = "mongodb://localhost:27017/"
client = pymongo.MongoClient(uri)
db = client.TwitterDB
