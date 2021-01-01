from pymongo import MongoClient
import operator
import json
########################################################################################################################
connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterDB
collections = db.collection_names()

########################################################################################################################
counter = 0
bigram_dictionary = dict()
trigram_dictionary = dict()
fourgram_dictionary = dict()
for collection in collections:
    tweets = db[collection].find(no_cursor_timeout=True).batch_size(10)
    if collection == 'vaccine_test4':
        for tweet in tweets:
            counter += 1
            print(counter)
            # bi_text = db[collection].find_one({'bigrams': tweet["bigrams"]})["bigrams"]
            bi_text = tweet["bigrams"]
            for bigram in bi_text:
                try:
                    bigram_key_value = bigram[0][0] + ' ' + bigram[0][1]
                    if bigram_key_value in bigram_dictionary:
                        bigram_dictionary[bigram_key_value] += bigram[1]
                    else:
                        bigram_dictionary[bigram_key_value] = bigram[1]
                except:
                    continue
            # tri_text = db[collection].find_one({'trigrams': tweet["trigrams"]})["trigrams"]
            tri_text = tweet["trigrams"]
            for trigram in tri_text:
                try:
                    trigram_key_value = trigram[0][0] + ' ' + trigram[0][1] + ' ' + trigram[0][2]
                    if trigram_key_value in trigram_dictionary:
                        trigram_dictionary[trigram_key_value] += trigram[1]
                    else:
                        trigram_dictionary[trigram_key_value] = trigram[1]
                except:
                    continue
            # four_text = db[collection].find_one({'fourgrams': tweet["fourgrams"]})["fourgrams"]
            four_text = tweet["fourgrams"]
            for fourgram in four_text:
                try:
                    fourgram_key_value = fourgram[0][0] + ' ' + fourgram[0][1] + ' ' + fourgram[0][2] + ' ' + \
                                         fourgram[0][3]
                    if fourgram_key_value in fourgram_dictionary:
                        fourgram_dictionary[fourgram_key_value] += fourgram[1]
                    else:
                        fourgram_dictionary[fourgram_key_value] = fourgram[1]
                except:
                    continue
            if (counter >= 1000) & (counter % 1000 == 0):
                # print("deleting..")
                bigram_dictionary = {key: val for key, val in bigram_dictionary.items() if val >= 2}
                trigram_dictionary = {key: val for key, val in trigram_dictionary.items() if val >= 2}
                fourgram_dictionary = {key: val for key, val in fourgram_dictionary.items() if val >= 2}
    bigram_dictionary = {key: val for key, val in bigram_dictionary.items() if val >= 2}
    trigram_dictionary = {key: val for key, val in trigram_dictionary.items() if val >= 2}
    fourgram_dictionary = {key: val for key, val in fourgram_dictionary.items() if val >= 2}


print("writing bigrams")
bigrams_file = dict(sorted(bigram_dictionary.items(), key=operator.itemgetter(1), reverse=True))
with open('Ngrams/bigrams_vaccine__final.txt', 'w') as file:
    file.write(json.dumps(bigrams_file))
print("writing trigrams")
trigrams_file = dict(sorted(trigram_dictionary.items(), key=operator.itemgetter(1), reverse=True))
with open('Ngrams/trigrams_vaccine_final.txt', 'w') as file:
    file.write(json.dumps(trigrams_file))
print("writing fourgrams")
fourgrams_file = dict(sorted(fourgram_dictionary.items(), key=operator.itemgetter(1), reverse=True))
with open('Ngrams/fourgrams_vaccine_final.txt', 'w') as file:
    file.write(json.dumps(fourgrams_file))
