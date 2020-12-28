import demoji
from pymongo import MongoClient
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from FileAlfaGama.data import *
from FileAlexia.tweet_location import *
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem import PorterStemmer
from datetime import datetime
from collections import Counter
from nltk import pos_tag
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

#####################################################################
connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterDB

collections = db.collection_names()

#####################################################################
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")


#####################################################################


def preprocessing(text):
    """
    :param txt: (string) for preprocessing
    :return: (list) tokens, (list) tokens preprocessed, (string) text preprocessed
    """
    #   We set all words to lower-case
    text_lowered = text.lower()

    #   We modified the contractions & split it into tokens
    tokens_no_contractions = [get_contractions(word) for word in text_lowered.split()]

    #   We removed the hashtag symbol and its content (e.g., #COVID19), @users, and URLs from the messages because the
    #       hashtag symbols or the URLs did not contribute to the message analysis.
    tokens_basic_pre = [token for token in tokens_no_contractions if not token.startswith('http')
                        if not token.startswith('#') if not token.startswith('@')]

    #   We removed all non-English characters (non-ASCII characters) because the study focused on the analysis of
    #       messages in English. (and numbers.)
    tokens_only_letters = list(filter(lambda ele: re.search("[a-zA-Z\s]+", ele) is not None, tokens_basic_pre))

    #   We deTokenize here in order to use RE more efficinetly
    text_deTokenized = TreebankWordDetokenizer().detokenize(tokens_only_letters)

    #   We use RE to remove any unwanted characters from the stings
    text_deTokenized = re.sub(r'[0-9]', '', text_deTokenized)
    text_deTokenized = re.sub(r'[!@#$%^&*();:"?.>,<`~-]', '', text_deTokenized)
    text_deTokenized = re.sub(r"[']", '', text_deTokenized)
    text_deTokenized = re.sub(r"[/]", ' ', text_deTokenized)
    text_deTokenized = re.sub(r"\t", " ", text_deTokenized)
    text_deTokenized = re.sub(r"'\s+\s+'", " ", text_deTokenized)
    text_deTokenized = re.sub(r" ️", "", text_deTokenized)

    #   We deleted all emoji with the help of demoji library
    emoji_to_delete = demoji.findall(text_deTokenized)
    for emoji in emoji_to_delete:
        text_deTokenized = re.sub(emoji, '', text_deTokenized)

    #   We tokenized again
    tokens_again = text_deTokenized.split()

    #   We removed stop_words
    final_stop_words = [x for x in stop_words if x not in ok_stop_words]
    tokens_no_stop_words = [w for w in tokens_again if w not in final_stop_words]

    #   We used WordNetLemmatizer from the nltk library as a final step
    lemmatizer = WordNetLemmatizer()
    # tokens_lemmatized = [lemmatizer.lemmatize(l) for l in tokens_no_stop_words]
    #   We also decided to use pos_tagging to enhance our lemmatization model
    tokens_lemmatized = []
    for word, tag in pos_tag(tokens_no_stop_words):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        tokens_lemmatized.append(lemmatizer.lemmatize(word, pos))

    #   We used PorterStemmer from the nltk library as a final step
    # ps = PorterStemmer()
    # tokens_stemmed = [ps.stem(w) for w in tokens_no_stop_words]

    # return a list of tokens without preprocessing, a list of tokens after all preprocessing pipeline
    #   and a string of full text preprocessed
    return text.split(), tokens_lemmatized, TreebankWordDetokenizer().detokenize(tokens_lemmatized)


#####################################################################


def update_preprocessed_column(f_tokens, clean_tokens, clean_text, date, country):
    """
    :param f_tokens: tokenized full_text to import in a new column in db
    :param clean_tokens: tokenized & cleaned full_text to import in a new column in db
    :param clean_text: cleaned full_text to import in a new column in db
    :return: None
    """
    if clean_text == '':
        #   delete
        db[collection].delete_one(
            {
                "id": tweet["id"]
            }
        )
    else:
        #   update new columns
        db[collection].update(
            {
                "id": tweet["id"]
            },
            {
                "$set": {
                    # "word2vec": aaa,
                    # "tfidf": bbb,
                    "tweet_date": date,
                    "location": country,
                    "tokens": f_tokens,
                    "tokens_preprocessed": clean_tokens,
                    "text_preprocessed": clean_text,
                    "bigrams": Counter(zip(clean_tokens, clean_tokens[1:])).most_common(),
                    "trigrams": Counter(zip(clean_tokens, clean_tokens[1:], clean_tokens[2:])).most_common(),
                    "fourgrams": Counter(zip(clean_tokens, clean_tokens[1:], clean_tokens[2:], clean_tokens[3:])).most_common(),
                    "bigrams_full": Counter(zip(f_tokens, f_tokens[1:])).most_common(),
                    "trigrams_full": Counter(zip(f_tokens, f_tokens[1:], f_tokens[2:])).most_common(),
                    "fourgrams_full": Counter(zip(f_tokens, f_tokens[1:], f_tokens[2:], f_tokens[3:])).most_common()
                }
            }
        )


#####################################################################
#   stuff for print at the end!
all_dates = []
total_tweets = collection.estimated_document_count()
tweets_loc_found_count = 0
tweet_index = 0

for collection in collections:
    tweets = db[collection].find().batch_size(10)
    if collection == 'vaccine':
        for tweet in tweets:
            #   update Date
            date = db[collection].find_one({'created_at': tweet["created_at"]})["created_at"]
            new_datetime = datetime.strftime(datetime.strptime(date, '%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d')
            #   update location
            tweet_country = getTweetLocation(tweet)
            #   preprocessed data + ngrams
            text = db[collection].find_one({'full_text': tweet["full_text"]})["full_text"]
            tokens, preprocessed_tokens, preprocessed_text = preprocessing(text)
            update_preprocessed_column(tokens, preprocessed_tokens, preprocessed_text, new_datetime, tweet_country)
            # prints for data
            print("Before: ")
            print(text)
            print("After: ")
            print(tokens)
            print(preprocessed_tokens)
            # prints for date
            all_dates.append(new_datetime)
            # prints for location
            tweet_index += 1
            if tweet_country:
                tweets_loc_found_count += 1

unique_dates_frequency = Counter(all_dates)
print(unique_dates_frequency.items())

print('Found country for ', tweets_loc_found_count, ' tweets from ', total_tweets, ' total tweets')
print('Percentage of tweets with country: ', tweets_loc_found_count / total_tweets * 100)

#####################################################################
