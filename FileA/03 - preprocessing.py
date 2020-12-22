import re
import pymongo
from pymongo import MongoClient
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from FileA.data import *
from nltk.tokenize.treebank import TreebankWordDetokenizer

connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterDB

collections = db.collection_names()

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")


def preprocessing(txt):
    """
    :param txt: (string) for preprocessing
    :return: (list) tokens, (list) tokens preprocessed, (string) text preprocessed
    """
    #   We tokenize the tweet
    tokens = txt.split()

    #   We set all words to lower-case
    tokens_lower = [x.lower() for x in tokens]

    #   We modified the contractions
    tokens_no_contractions = [get_contractions(word) for word in tokens_lower]

    #   We removed the hashtag symbol and its content (e.g., #COVID19), @users, and URLs from the messages because the
    #       hashtag symbols or the URLs did not contribute to the message analysis.
    tokens_basic_pre = [token for token in tokens_no_contractions if not token.startswith('http')
                        if not token.startswith('#') if not token.startswith('@')]

    #   We removed all non-English characters (non-ASCII characters) because the study focused on the analysis of
    #       messages in English. (and numbers.)
    tokens_only_letters = list(filter(lambda ele: re.search("[a-zA-Z\s]+", ele) is not None, tokens_basic_pre))

    #   We also removed special characters, punctuations.
    tokens_no_specials = [re.sub('[^A-Za-z0-9]+', '', i) for i in tokens_only_letters]

    #   We removed repeated words. For example, sooooo terrified was converted to so terrified.
    tokens_no_dragging = tokens_no_specials  # (in progress ... )

    #   We removed stop_words
    tokens_no_stop_words = [w for w in tokens_no_dragging if not w in stop_words]

    # return a list of tokens without preprocessing, a list of tokens after all preprocessing pipeline
    #   and a string of full text preprocessed
    return tokens, tokens_no_stop_words, TreebankWordDetokenizer().detokenize(tokens_no_stop_words)



def update_preprocessed_column(tokens, clean_tokens, text):
    """
    :param tokens: import tokenized full_text to import in a new column in db
    :param clean_tokens: import tokenized & cleaned full_text to import in a new column in db
    :return: None
    """
    db[collection].update(
        {
            "full_text": tweet["full_text"]
        },
        {
            "$set": {
                "tokens": tokens,
                "tokens_preprocessed": clean_tokens,
                "text_preprocessed": text
            }
        }
    )


for collection in collections:
    tweets = db[collection].find().batch_size(10)
    if collection == 'vaccine_test':
        for tweet in tweets:
            text = db[collection].find_one({'full_text': tweet["full_text"]})["full_text"]
            print("Before: ")
            print(text)
            print("After: ")
            tokens, preprocessed_tokens, preprocessed_text = preprocessing(text)
            update_preprocessed_column(tokens, preprocessed_tokens, preprocessed_text)
            print(tokens)
            print(preprocessed_tokens)
