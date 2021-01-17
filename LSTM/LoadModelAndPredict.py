import pickle
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from pymongo import MongoClient


MAX_LENGTH = 34
EMBEDDING_DIM = 100
folder = 'BiLSTM-glove-binary-trainableTrue-noPrecision'


# MongoDB
#----------------------------------
connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterDB
collection = db['temp2']
#----------------------------------



# Load LSTM model
#----------------------------------
loaded_model = load_model('models/'+ folder +'/BiLSTM_glove_model.h5')
loaded_model.summary()
#----------------------------------



# Load tokenizer
#----------------------------------
with open('models/'+ folder +'/BiLSTM_glove_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
#----------------------------------



# Predict label on a custom str for test
#----------------------------------
#b = tokenizer.texts_to_sequences(['bio terrorist endanger public lock aid air bear would tolerate mass gather claim aid hoax'])
#b = pad_sequences(b, maxlen=MAX_LENGTH)
#print(loaded_model.predict(b))
#print(loaded_model.predict_classes(b)[0])
#----------------------------------



# Predict label on our tweets
#----------------------------------
tweets = collection.find().batch_size(10)
tweet_index = 0
for tweet in tweets:
    print(tweet['text_preprocessed'])
    b = tokenizer.texts_to_sequences([tweet['text_preprocessed']])
    b = pad_sequences(b, maxlen=MAX_LENGTH)

    print(loaded_model.predict(b))
    label = loaded_model.predict_classes(b)
    label = label.flat[0]

    if label == 1:
        label_str = 'positive'
    elif label == 0:
        label_str = 'negative'
    else:
        print("Problem! Neither positive nor negative!")

    print(label)
    print(label_str)

    collection.update(
        {
            "_id": tweet["_id"]
        },
        {
            "$set": {
                "BiLSTM_label": label_str,
            }
        }
    )
    tweet_index += 1
    print('[Sentiment Update]', tweet['id'], '[' + str(tweet_index) + '/' + str(collection.estimated_document_count()) + ']\n')
#----------------------------------
