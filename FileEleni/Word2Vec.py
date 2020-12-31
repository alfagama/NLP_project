from pymongo import MongoClient
from gensim.models.word2vec import Word2Vec
# Save Model Using joblib
import joblib
import time


connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterCovidDB
collection = db['quarantine']

cursor = collection.find({}, {"label":1, "filtered_text":1, "tokens_preprocessed":1, "id":1, "_id":0}).batch_size(20).limit(100000)
documents = []

errorIds = []
start = time.time()
print("hello")

for document in cursor:
    try:
        for word in document["tokens_preprocessed"]:
            documents.append(word)
    except (KeyError):
        #print("I have a keyError in ", document["id"])
        errorIds.append(document["id"])
        pass

end = time.time()
print(end - start)

#print(documents)
if len(errorIds) > 0:
    print("I have a keyError in ", errorIds)
    print(len(errorIds))


#### SAVE THE WORD2VEC MODEL

word2VecModel = Word2Vec([documents], window=4, size=300, workers=10, min_count=1)
word2VecModel.build_vocab([documents], update=True)
word2VecModel.train([documents], total_examples=1, epochs=1)
filename = 'Word2Vec_model.sav'
joblib.dump(word2VecModel, filename)



### Load the model from disk
filename = 'Word2Vec_model.sav'
loaded_model = joblib.load(filename)

#print(loaded_model["covid"])
print("")
#print(loaded_model["covid", "interview"])


cursor = collection.find({}, {"label":1, "filtered_text":1, "id":1, "tokens_preprocessed":1, "_id":0}).batch_size(20).limit(10000)
for document in cursor:
    try:
        print(document["tokens_preprocessed"])
        text = document["tokens_preprocessed"]
        collection.update(
            {
                "id": document["id"]
            },
            {
                "$set": {
                    "word2vec": loaded_model[text].tolist()
                }
            }
        )
    except (KeyError):
        print("I have a keyError in ", document["id"])
        errorIds.append(document["id"])
        pass

