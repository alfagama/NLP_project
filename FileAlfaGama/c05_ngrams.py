from pymongo import MongoClient
import operator
import json

########################################################################################################################
connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterDB
collections = db.collection_names()

########################################################################################################################
#   bi-gram
bigram_dictionary = dict()
for collection in collections:
    tweets = db[collection].find().batch_size(10)
    if collection == 'vaccine_test3':
        for tweet in tweets:
            bi_text = db[collection].find_one({'bigrams': tweet["bigrams"]})["bigrams"]
            # bi_text = db[collection].find_one({'bigrams_full': tweet["bigrams_full"]})["bigrams_full"]
            for bigram in bi_text:
                try:
                    bigram_key_value = bigram[0][0] + ' ' + bigram[0][1]
                    if bigram_key_value in bigram_dictionary:
                        bigram_dictionary[bigram_key_value] += bigram[1]
                    else:
                        bigram_dictionary[bigram_key_value] = bigram[1]
                except:
                    continue

# print(dict(sorted(bigram_dictionary.items(), key=lambda item: item[1], reverse=True)))
# print(dict(sorted(bigram_dictionary.items(), key=operator.itemgetter(1), reverse=True)))
bigrams_file = dict(sorted(bigram_dictionary.items(), key=operator.itemgetter(1), reverse=True))
with open('Ngrams/bigrams_file3.txt', 'w') as file:
    file.write(json.dumps(bigrams_file))

########################################################################################################################
#   tri-gram
trigram_dictionary = dict()
for collection in collections:
    tweets = db[collection].find().batch_size(10)
    if collection == 'vaccine_test3':
        for tweet in tweets:
            tri_text = db[collection].find_one({'trigrams': tweet["trigrams"]})["trigrams"]
            # tri_text = db[collection].find_one({'trigrams_full': tweet["trigrams_full"]})["trigrams_full"]
            for trigram in tri_text:
                try:
                    trigram_key_value = trigram[0][0] + ' ' + trigram[0][1] + ' ' + trigram[0][2]
                    if trigram_key_value in trigram_dictionary:
                        trigram_dictionary[trigram_key_value] += trigram[1]
                    else:
                        trigram_dictionary[trigram_key_value] = trigram[1]
                except:
                    continue

# print(dict(sorted(trigram_dictionary.items(), key=lambda item: item[1], reverse=True)))
# print(dict(sorted(trigram_dictionary.items(), key=operator.itemgetter(1), reverse=True)))
trigrams_file = dict(sorted(trigram_dictionary.items(), key=operator.itemgetter(1), reverse=True))
with open('Ngrams/trigrams_file3.txt', 'w') as file:
    file.write(json.dumps(trigrams_file))

########################################################################################################################
#  four-gram
fourgram_dictionary = dict()
for collection in collections:
    tweets = db[collection].find().batch_size(10)
    if collection == 'vaccine_test3':
        for tweet in tweets:
            four_text = db[collection].find_one({'fourgrams': tweet["fourgrams"]})["fourgrams"]
            # four_text = db[collection].find_one({'fourgrams_full': tweet["fourgrams_full"]})["fourgrams_full"]
            for fourgram in four_text:
                try:
                    fourgram_key_value = fourgram[0][0] + ' ' + fourgram[0][1] + ' ' + fourgram[0][2] + ' ' + fourgram[0][3]
                    if fourgram_key_value in fourgram_dictionary:
                        fourgram_dictionary[fourgram_key_value] += fourgram[1]
                    else:
                        fourgram_dictionary[fourgram_key_value] = fourgram[1]
                except:
                    continue

# print(dict(sorted(fourgram_dictionary.items(), key=lambda item: item[1], reverse=True)))
# pprint.pprint(dict(sorted(fourgram_dictionary.items(), key=operator.itemgetter(1), reverse=True)))
fourgrams_file = dict(sorted(fourgram_dictionary.items(), key=operator.itemgetter(1), reverse=True))
with open('Ngrams/fourgrams_file3.txt', 'w') as file:
    file.write(json.dumps(fourgrams_file))

########################################################################################################################
