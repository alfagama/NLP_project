#   Imports
#   ##################################################################################################
import ktrain
from ktrain import text
import pymongo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import keras
import numpy
import time
from sklearn import metrics

start_time = time.time()
#   Vars
#   ##################################################################################################
COLLECTIONAME = 'sentiment140'
uri = "mongodb://localhost:27017/"
client = pymongo.MongoClient(uri)
db = client.TwitterDB
collection = db[COLLECTIONAME]
logdir = "logs/"
checkpoint_path = "weights_albert/"
checkpoint_dir = os.path.dirname(checkpoint_path)

#   Get collection in Df
#   ##################################################################################################
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
tweets = collection.find({}, {'text_preprocessed': 1, 'label': 1,
                              '_id': 0})  # get only 'text_preprocessed' and 'label' from all records
data = pd.DataFrame(list(tweets))
data = shuffle(data)

#   Replace positive from 4 to 1
#   ##################################################################################################
data['label'] = data['label'].replace('4', '1')  # replace 4 (meaning positive) to 1
label_list = list(set(data["label"]))
print("Label List: ", label_list)
print(data['label'].value_counts())

#   Train / Test Split
#   ##################################################################################################
x_train, x_test, y_train, y_test = train_test_split(data['text_preprocessed'],
                                                    data['label'],
                                                    test_size=0.2,
                                                    random_state=11)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(y_test.value_counts())
x_test = tuple(x_test)
x_test = numpy.asarray(x_test)

#   Train / Test Split
#   ##################################################################################################
sentence_list = [[s for s in list.split()] for list in x_train]  # convert to list of lists of words
max_length = len(max(sentence_list, key=len))  # find the max length of list
print("Max length: ", max_length)

#   Model
#   ##################################################################################################
num_of_epochs = 10
MODEL_NAME = 'albert-base-v2'

t = text.Transformer(MODEL_NAME,
                     maxlen=max_length,
                     class_names=label_list)

x_t, x_val, y_t, y_val = train_test_split(x_train,
                                          y_train,
                                          test_size=0.1,
                                          random_state=11)

trn = t.preprocess_train(x_t.to_numpy(), y_t.to_numpy())
val = t.preprocess_test(x_val.to_numpy(), y_val.to_numpy())

model = t.get_classifier()

tbCallBack = keras.callbacks.TensorBoard(log_dir=logdir,
                                         write_graph=True,
                                         write_images=True)

learner = ktrain.get_learner(model,
                             train_data=trn,
                             val_data=val,
                             batch_size=6)

learner.fit_onecycle(3e-5,
                     int(num_of_epochs),
                     checkpoint_folder=checkpoint_path,
                     callbacks=[tbCallBack])

model.summary()

print("--- %s seconds ---" % (time.time() - start_time))
predictor = ktrain.get_predictor(learner.model, preproc=t)
predictions = predictor.predict(x_test)
print(predictions)

print("\nEvaluation on test data using Sklearn metrics:")
print("Accuracy: %.6f" % metrics.accuracy_score(y_test, predictions))
print("Precision: %.6f" % metrics.precision_score(y_test, predictions, average='macro'))
print("Recall: %.6f" % metrics.recall_score(y_test, predictions, average='macro'))
print("F1: %.6f \n" % metrics.f1_score(y_test, predictions, average='macro'))
