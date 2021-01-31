from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from numpy import array
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from pymongo import MongoClient
import pickle
import re
import nltk
from nltk.corpus import stopwords
from keras.preprocessing.text import one_hot
from keras.models import Sequential
from sklearn.utils import shuffle
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, Input
from keras.layers import Conv1D
from keras.layers import GlobalMaxPooling1D, SpatialDropout1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import MaxPooling1D
from keras import regularizers
from numpy import array
from numpy import asarray
from numpy import zeros
from sklearn import metrics
from keras.utils.np_utils import to_categorical
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


'''
COLLECTIONAME = 'Sent140'


#MongoDB
#----------------------------------
uri = "mongodb://localhost:27017/"
client = MongoClient(uri)
db = client.TwitterCovidDB
collection = db[COLLECTIONAME]
#----------------------------------

#Create the dataframe
#----------------------------------
tweets = collection.find({}, {'text_preprocessed':1, 'label':1 , '_id':0}) #get only 'text_preprocessed' and 'label' from all records
data = pd.DataFrame(list(tweets))
data = shuffle(data)
#print(data.head(997))
#----------------------------------



# Convert positive from 4 to 1
#----------------------------------
data['label'] = data['label'].replace('4','1') #replace 4 (meaning positive) to 1
data['label'] = data['label'].replace(0,'0')
#----------------------------------

'''



# LOAD THE SENTIMENT140 FROM CSV
#----------------------------------------------------------------------------

df = pd.read_csv("sentiment140.csv")
#df = pd.concat([df.query("label==0").sample(50), df.query("label==4").sample(50)])
df = df.drop_duplicates(subset=['text_preprocessed'])
X = list(df['text_preprocessed'])

y = df['label']
y = np.array(list(map(lambda x: "1" if x==4 else "0", y)))

print("Size of X: ", len(X))
print("Size of y: ", len(y))
#----------------------------------------------------------------------------


# Split in train and test
#----------------------------------------------------------------------------
#X_train, X_test, y_train, y_test = train_test_split(data['text_preprocessed'], data['label'], test_size=0.20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
#----------------------------------------------------------------------------


# Convert label to categorical
#----------------------------------------------------------------------------
# When working with categorical data, we don’t want to leave it as integers because the model will interpreted the samples
# with a higher number as having more significance. to_categorical is a quick way of encoding the data.
#[0. 1.] -> positive -> 4 (or 1)
#[1. 0.] -> negative -> 0
'''
Y_test_classes_for_evaluation = y_test.astype(int)
Y_train = to_categorical(y_train, num_classes=2)
Y_test = to_categorical(y_test, num_classes=2)
'''
#----------------------------------------------------------------------------



# Count total number of distinct words of train set
#----------------------------------------------------------------------------

train_words_counter = Counter(word for sentence in list(X_train) for word in sentence.split())
#train_words_count = sorted(train_words_counter.items(), key=lambda kv: kv[1])
train_words_count = len(train_words_counter)
print("Total number of distinct words of train set: ", train_words_count)

#----------------------------------------------------------------------------



# Compute max number of words of sentences of train set
# Max_length is used for pad_sequences (we don't want to 'cut' any sentence, we keep all sentences with all their words)
#----------------------------------------------------------------------------

sentence_list = [[s for s in list.split()] for list in X_train] #convert to list of lists of words
maxLength = len(max(sentence_list, key=len)) # find the max length of list
print("Max length: ", maxLength)

#----------------------------------------------------------------------------



# Tokenizer and find the vocabulary (#of distinct words)
#----------------------------------------------------------------------------
num_of_words = round(train_words_count/10)
print("Num_words: ", num_of_words)

#The words not included in the vocabulary are replaced by <UKN>
tokenizer = Tokenizer(num_words= num_of_words, split=' ', oov_token="<UKN>")
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary size is: ", vocab_size)

#Padding sequences
X_train = pad_sequences(X_train, padding='post', maxlen=maxLength)
X_test = pad_sequences(X_test, padding='post', maxlen=maxLength)

#----------------------------------------------------------------------------



#### Create embeddings using GloVe
#----------------------------------------------------------------------------

embeddings_dictionary = dict()
#glove_file = open('glove.6B/glove.6B.100d.txt', encoding="utf8")
glove_file = open('glove.twitter.27B/glove.twitter.27B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
#----------------------------------------------------------------------------



#### Create the CNN model
#----------------------------------------------------------------------------
batch_size = 128
epochs = 10
model = Sequential()
#weights=[embedding_matrix],
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxLength, trainable=True)
model.add(embedding_layer)
model.add(SpatialDropout1D(0.5))
model.add(Conv1D(64, 20, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

print(model.summary())
#----------------------------------------------------------------------------



### Fit and evaluate the model
#----------------------------------------------------------------------------
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, shuffle=True, validation_split=0.2)
#----------------------------------------------------------------------------



# Save the model
#----------------------------------------------------------------------------
model.save("withGlove128-100d-0.5d-64-50TrainableTrueFinal.model")

filename = 'withGlove128-100d-0.5d-64-50TrainableTrueFinal.sav'
joblib.dump(model, filename)
#----------------------------------------------------------------------------


# Evaluation with Keras
#----------------------------------------------------------------------------
#loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=1, batch_size=batch_size)
loss, accuracy, f1_score, precision, recall= model.evaluate(X_test, y_test, verbose=1, batch_size=batch_size)
#print(model.metrics_names)
print("Evaluation on test data using Keras metrics:")
print("Loss: %.6f" % (loss))
print("Accuracy: %.6f" % (accuracy))
print("Precision: %.6f" % (precision))
print("Recall: %.6f" % (recall))
print("F1: %.6f" % (f1_score))
#----------------------------------------------------------------------------


#Evaluation with Sklearn
#----------------------------------------------------------------------------
y_pred = model.predict_classes(X_test, verbose=1)

print("\nEvaluation on test data using Sklearn metrics:")
print("Accuracy: %.6f" % metrics.accuracy_score(y_test.astype(int), y_pred))
print("Precision: %.6f" % metrics.precision_score(y_test.astype(int), y_pred, average='macro'))#, labels=np.unique(y_pred)))#, labels=np.unique(y_predicted)
print("Recall: %.6f" % metrics.recall_score(y_test.astype(int), y_pred, average='macro'))
print("F1: %.6f \n" % metrics.f1_score(y_test.astype(int), y_pred, average='macro'))
#----------------------------------------------------------------------------


# Plot accuracy and loss
#----------------------------------------------------------------------------
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#----------------------------------------------------------------------------


# Save tokenizer
#----------------------------------------------------------------------------
print("Saving tokenizer to disk")
with open('CNNwithGlove_128-100d-0.5d-64-50TrainableTrueFinal.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
#----------------------------------------------------------------------------
