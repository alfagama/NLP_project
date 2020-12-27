import nltk
from pymongo import MongoClient
nltk.download('twitter_samples')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import twitter_samples
from collections import Counter
from data import *
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.treebank import TreebankWordDetokenizer

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

    #   We replaced emoticons with words that express similar feeling
    tokens_no_empticons = [emoticon_translation(word) for word in tokens_no_contractions]

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
    tokens_no_dragging = tokens_no_specials
    # tokens_no_dragging = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tokens_no_specials))

    #   We removed stop_words
    final_stop_words = [x for x in stop_words if x not in ok_stop_words]
    tokens_no_stop_words = [w for w in tokens_no_dragging if w not in final_stop_words]

    #   We used WordNetLemmatizer from the nltk library as a final step
    """"
    lemmatizer = WordNetLemmatizer()
    tokens_lemmatized = [lemmatizer.lemmatize(l) for l in tokens_no_stop_words]print("BEFORE")
    print(tokens_lemmatized)
    """
    lemmatizer = WordNetLemmatizer()
    tokens_lemmatized = []
    for word, tag in pos_tag(tokens_no_stop_words):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        tokens_lemmatized.append(lemmatizer.lemmatize(word, pos))
    """
    print("AFTER")
    print(tokens_lemmatized)
    """
    #   We used PorterStemmer from the nltk library as a final step
    # ps = PorterStemmer()
    # tokens_stemmed = [ps.stem(w) for w in tokens_no_stop_words]

    # return a list of tokens without preprocessing, a list of tokens after all preprocessing pipeline
    #   and a string of full text preprocessed
    return tokens, tokens_lemmatized, TreebankWordDetokenizer().detokenize(tokens_lemmatized)


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


def loadToDB(tweet,label, collection):
    '''
        loads data to MongoDB
    '''
    tokens, preprocessed_tokens, preprocessed_text = preprocessing(tweet)
    print(tokens)
    print(preprocessed_tokens)
    print(tweet)
    document = {
        "full_text": tweet,
        "label": int(label),
        "text_preprocessed": preprocessed_text,
        "tokens": tokens,
        "tokens_preprocessed": preprocessed_tokens,
        "bigrams": Counter(zip(preprocessed_tokens, preprocessed_tokens[1:])).most_common(),
        "trigrams": Counter(zip(preprocessed_tokens, preprocessed_tokens[1:], preprocessed_tokens[2:])).most_common(),
        "fourgrams": Counter(zip(preprocessed_tokens, preprocessed_tokens[1:], preprocessed_tokens[2:], preprocessed_tokens[3:])).most_common(),
        "bigrams_full": Counter(zip(tokens, tokens[1:])).most_common(),
        "trigrams_full": Counter(zip(tokens, tokens[1:], tokens[2:])).most_common(),
        "fourgrams_full": Counter(zip(tokens, tokens[1:], tokens[2:], tokens[3:])).most_common()
    }
    print(document)
    try:
        collection.insert_one(document)
    except DuplicateKeyError as e:
        print('Duplicate tweet detected', document)



uri = "mongodb://localhost:27017/"
client = MongoClient(uri)
db = client.TwitterCovidDB
CollectionName = 'nltkSamples'
collection = db[CollectionName]


positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')


stop_words = set(stopwords.words('english'))

for tweet in positive_tweets:
    loadToDB(tweet,1,collection)

for tweet in negative_tweets:
    loadToDB(tweet,0,collection)

"""
loadToDB('specified specifies specify helen run ran runs hello members member did do',1,collection)
"""
#all_pos_words = get_all_words(positive_cleaned_tokens_list)