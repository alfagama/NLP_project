from pymongo import MongoClient
from itertools import islice

connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterDB
collections = db.collection_names()

#   tri-gram
fourgram_dictionary = dict()
for collection in collections:
    tweets = db[collection].find().batch_size(10)
    if collection == 'vaccine_test':
        for tweet in tweets:
            four_text = db[collection].find_one({'fourgrams': tweet["fourgrams"]})["fourgrams"]
            for fourgram in four_text:
                try:
                    fourgram_key_value = fourgram[0][0] + ' ' + fourgram[0][1] + ' ' + fourgram[0][2] + ' ' + fourgram[0][3]
                    # print(fourgram_key_value)
                    # print(fourgram[1])
                    if fourgram_key_value in fourgram_dictionary:
                        fourgram_dictionary[fourgram_key_value] += fourgram[1]
                    else:
                        fourgram_dictionary[fourgram_key_value] = fourgram[1]
                except:
                    continue

print(dict(sorted(fourgram_dictionary.items(), key=lambda item: item[1], reverse=True)))
# print(list(islice(fourgram_dictionary, 10)))
