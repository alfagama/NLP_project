from pymongo import MongoClient

connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterDB
collections = db.collection_names()

#   tri-gram
trigram_dictionary = dict()
for collection in collections:
    tweets = db[collection].find().batch_size(10)
    if collection == 'vaccine_test':
        for tweet in tweets:
            tri_text = db[collection].find_one({'trigrams': tweet["trigrams"]})["trigrams"]
            for trigram in tri_text:
                try:
                    trigram_key_value = trigram[0][0] + ' ' + trigram[0][1] + ' ' + trigram[0][2]
                    if trigram_key_value in trigram_dictionary:
                        trigram_dictionary[trigram_key_value] += trigram[1]
                    else:
                        trigram_dictionary[trigram_key_value] = trigram[1]
                except:
                    continue

print(dict(sorted(trigram_dictionary.items(), key=lambda item: item[1], reverse=True)))

# lista = list(dict(sorted(trigram_dictionary.items(), key=lambda item: item[1], reverse=True)))
# print(lista)
# for x in range(0, 5):
#     print(lista[x])
#     print(lista[x][1])
