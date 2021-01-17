import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

names = ['_id','location','text_preprocessed','tweet_date']
df = pd.read_csv(
    'train/quarantine.csv',
    names=names,
    sep=',',
    header=1,  # no header, alternative header = header_col
    index_col=None,  # no index, alternative header = index_row
    skiprows=1  # how many rows to skip / not include in read_csv
)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(df.head())

MAX_LENGTH = 34
EMBEDDING_DIM = 100
folder = ''

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

X_test = df['text_preprocessed'].tolist()

b = tokenizer.texts_to_sequences(X_test)
b = pad_sequences(b, maxlen=MAX_LENGTH)

y_pred = loaded_model.predict(b, verbose=0)
print(y_pred)
y_pred_str = [("1" if i >= 0.500000000 else "0") for i in y_pred]

df['label'] = y_pred_str
df.to_csv('quarantine_with_label.csv')
