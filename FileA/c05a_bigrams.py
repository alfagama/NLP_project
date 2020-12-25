from pymongo import MongoClient

connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterDB
collections = db.collection_names()

#   bi-gram
bigram_dictionary = dict()
for collection in collections:
    tweets = db[collection].find().batch_size(10)
    if collection == 'vaccine_test':
        for tweet in tweets:
            bi_text = db[collection].find_one({'bigrams': tweet["bigrams"]})["bigrams"]
            for bigram in bi_text:
                try:
                    bigram_key_value = bigram[0][0] + ' ' + bigram[0][1]
                    if bigram_key_value in bigram_dictionary:
                        bigram_dictionary[bigram_key_value] += bigram[1]
                    else:
                        bigram_dictionary[bigram_key_value] = bigram[1]
                except:
                    continue

print(dict(sorted(bigram_dictionary.items(), key=lambda item: item[1], reverse=True)))

