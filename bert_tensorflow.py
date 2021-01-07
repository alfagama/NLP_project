import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
from sklearn.model_selection import train_test_split

model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print(model.summary())

df = pd.read_csv("Data/sentiment140.csv",
                 sep=',',
                 names=['_id', 'label', 'id', 'tweet_text', 'text_preprocessed', 'tokens_preprocessed'],
                 index_col=None,
                 skiprows=0
                 )

df = df.drop(['_id', 'tweet_text', 'tokens_preprocessed'], axis=1)
df = df.rename(columns={'label': 'LABEL_COLUMN', 'text_preprocessed': 'DATA_COLUMN'})
df.set_index('id', inplace=True)
print(df.head())

train, test = train_test_split(df, test_size=0.2, random_state=11)
# print(train.head())
# print(test.head())
# print(len(train))
# print(len(test))

df_test = pd.read_csv("Data/vaccine_db.csv",
                      sep=',',
                      names=['_id', 'id', 'text_preprocessed', 'tweet_date', 'country',
                             'textblob_preprocessed_label', 'vader_preprocessed_label'],
                      index_col=None,
                      skiprows=0
                      )

df_test = df_test.rename(columns={'text_preprocessed': 'DATA_COLUMN'})
df_test.set_index('id', inplace=True)
print(df_test.head())


def convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN):
    train_InputExamples = train.apply(
        lambda x: InputExample(guid=None,  # Globally unique ID for bookkeeping, unused in this case
                               text_a=x[DATA_COLUMN],
                               text_b=None,
                               label=x[LABEL_COLUMN]), axis=1)

    validation_InputExamples = test.apply(
        lambda x: InputExample(guid=None,  # Globally unique ID for bookkeeping, unused in this case
                               text_a=x[DATA_COLUMN],
                               text_b=None,
                               label=x[LABEL_COLUMN]), axis=1)

    return train_InputExamples, validation_InputExamples


def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = []  # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length,  # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True,  # pads to the right by default # CHECK THIS for pad_to_max_length
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
                                                     input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


DATA_COLUMN = 'DATA_COLUMN'
LABEL_COLUMN = 'LABEL_COLUMN'

train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN)

train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)

validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(32)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5,
                                                 epsilon=1e-08,
                                                 clipnorm=1.0),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

model.fit(train_data,
          epochs=5,
          validation_data=validation_data)

dataframe_to_predict = df_test['DATA_COLUMN'].astype(str).values.tolist()
print(dataframe_to_predict)

tf_batch = tokenizer(dataframe_to_predict,
                     max_length=128,
                     padding=True,
                     truncation=True,
                     return_tensors='tf')
tf_outputs = model(tf_batch)
tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
labels = ['0', '1']
label = tf.argmax(tf_predictions, axis=1)
label = label.numpy()
# for i in range(len(dataframe_to_predict)):
#   print(dataframe_to_predict[i], ": \n", labels[label[i]])
df_test['bert_label'] = label
df_test.to_csv(r'bert_labeled_full.csv', index=False, header=True)
