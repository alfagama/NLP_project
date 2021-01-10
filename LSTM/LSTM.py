import keras
import pymongo
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from collections import Counter
import pickle


COLLECTIONAME = 'sentiment140'


#MongoDB
#----------------------------------
uri = "mongodb://localhost:27017/"
client = pymongo.MongoClient(uri)
db = client.TwitterDB
collection = db[COLLECTIONAME]
#----------------------------------



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



#Create the dataframe
#----------------------------------
tweets = collection.find({}, {'text_preprocessed':1, 'label':1 , '_id':0}) #get only 'text_preprocessed' and 'label' from all records
data = pd.DataFrame(list(tweets))
data = shuffle(data)
#print(data.head(997))
#----------------------------------



# Convert label to categorical
#----------------------------------
data['label'] = data['label'].replace('4','1') #replace 4 (meaning positive) to 1

# When working with categorical data, we don’t want to leave it as integers because the model will interpreted the samples
# with a higher number as having more significance. to_categorical is a quick way of encoding the data.
#[0. 1.] -> positive -> 4
#[1. 0.] -> negative -> 0
Y = to_categorical(data['label'], num_classes=2)
#----------------------------------



# Train test split
#----------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(data['text_preprocessed'], Y, test_size=0.2, random_state=42)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)
#----------------------------------



# Count total number of distinct words
#----------------------------------
train_words_counter = Counter(word for sentence in list(X_train) for word in sentence.split())
#train_words_count = sorted(train_words_counter.items(), key=lambda kv: kv[1])
train_words_count = len(train_words_counter)
print("Total number of distinct words of train set: ", train_words_count)
#----------------------------------



# Vectorize with Tokenizer
#----------------------------------
#Should we fit on train data only or in our whole dataset (including our 3 collections)
#https://stackoverflow.com/questions/54891464/is-it-better-to-keras-fit-to-text-on-the-entire-x-data-or-just-the-train-data
#https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/
#https://towardsdatascience.com/machine-learning-recurrent-neural-networks-and-long-short-term-memory-lstm-python-keras-example-86001ceaaebc
#!!We fit on train data only and we transform text data and our collections!!!

# In specifying num_words, only the most common num_words-1 words will be kept. Should we keep all words?
# When keeping all words, the model overfits , so we keep a percentage of total number of words
# Maybe not all words are meaningfull in extracting sentiment
# Through experimentation we came to the conclusion that keepin the top 10% of total words leads to good results (no overfit)
num_of_words = round(train_words_count/10)
print("Num_words: ", num_of_words)

#The words not included in the vocabulary are replaced by <UKN>
tokenizer = Tokenizer(num_words= num_of_words, split=' ', oov_token="<UKN>")
tokenizer.fit_on_texts(X_train)

#print all words
#x = tokenizer.word_docs
#sorted_x = sorted(x.items(), key=lambda kv: kv[1])
#print(sorted_x)
#print("Total number of unique words: ", len(tokenizer.word_docs))


X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

phrase_len = data['text_preprocessed'].apply(lambda p: len(p.split(' '))) # compute the length of each phrase of each tweet
max_length = phrase_len.max() # find the max length
print("Max length: ", max_length)

# Convert to a sequence. Used to ensure that all phrases are the same length.
# Sequences that are shorter than maxlen are padded with value (0 by default)
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)
#----------------------------------



# LSTM model
#----------------------------------
input_dim = len(tokenizer.word_index) + 1 #+ 1 because of reserving padding (index zero)
#print(len(tokenizer.word_index))
#print(tokenizer.word_docs)

embed_dim = 20
lstm_out = 50

model = Sequential()
model.add(Embedding(input_dim=input_dim, output_dim=embed_dim, input_length=X_train.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.6, recurrent_dropout=0.6))
model.add(Dense(30, activation='sigmoid')) #sigmoid for binary classification, softmax for multiclass classifictaion
model.add(Dense(2, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='Adam', metrics=['accuracy']) #binary_crossentropy for binary classification, categorical_crossentropy for multiclass
print(model.summary())
#----------------------------------



# Fit model
#----------------------------------
batch_size = 32
checkpoint1 = ModelCheckpoint("weights/LSTM_best_model1.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto', period=1,save_weights_only=False)
history = model.fit(X_train, Y_train, epochs=8, validation_split=0.2, callbacks=[checkpoint1], batch_size=batch_size) #validation_data=(X_test, Y_test),
#----------------------------------



# Evaluate model
#----------------------------------
score, acc = model.evaluate(X_test, Y_test, verbose=1, batch_size=batch_size)
print("Score: %.6f" % (score))
print("Accuracy: %.6f" % (acc))
#----------------------------------



# Predict label
#----------------------------------
'''b = tokenizer.texts_to_sequences(['bio terrorist endanger public lock aid air bear would tolerate mass gather claim aid hoax'])
b = pad_sequences(b, maxlen=max_length)
print(model.predict(b))'''
#----------------------------------



# Save LSTM model
#----------------------------------
print("Saving LSTM model to disk")
model.save('models/LSTM_model.h5')
#----------------------------------



# Save tokenizer
#----------------------------------
print("Saving tokenizer to disk")
with open('models/LSTM_tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
#----------------------------------



# Plot accuracy and loss
#----------------------------------
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("plots/LSTM_accuracy.png")
plt.show()


# summarize history for loss
# If they to depart consistently, it might be a sign to stop training at an earlier epoch.
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("plots/LSTM_loss.png")
plt.show()
#----------------------------------




