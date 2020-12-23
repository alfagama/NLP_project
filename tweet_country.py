import pymongo
import geograpy
import re


#MongoDB
uri = "mongodb://localhost:27017/"
client = pymongo.MongoClient(uri)
db = client.TwitterDB
CollectionName = 'covidtemp'
collection = db[CollectionName]


def recognizeSpecificCountries(text):
    '''
    Gets the location if the text contains abbreviations or words like 'britain' 'england' that are not recognized by geograpy
    :param text: location text of tweet
    :return:
    '''

    #what happens is a tweet has more than one countries e.g. 'us | uk | eu' !!!!!!!!!!!!!!!!!!!!!!!!! todo
    text = text.lower()
    text_clean = re.sub('[^a-zA-Z\s]+', '', text)  # remove special characters, keep only the letters
    tokens = text_clean.split() #tokenize

    #print(text)
    #print(tokens)

    if 'usa' in tokens:
        return 'United States'

    if 'us' in tokens:
        return 'United States'

    if 'u.s.' in tokens:
        return 'United States'

    if 'new york city' in tokens:
        return 'United States'

    if 'new york' in tokens:
        return 'United States'

    if 'ny' in tokens:
        return 'United States'

    if 'uk' in tokens:
        return 'United Kingdom'

    if 'britain' in tokens:
        return 'United Kingdom'

    if 'england' in tokens: #England is regarded as a city in United States by geograpy!!
        return 'United Kingdom'

    #if 'London' in tokens: #There is a city named 'London' in United States and Canada, but most likely it refers to United Kingdom (sure??????) todo
        #return 'United Kingdom'

    if 'au' in tokens:
        return 'Australia'

    return None


def getTweetCountry(tweet):
    place = tweet['place']
    # geo = tweet['geo'] #no 'geo' found in tweets
    # coordinates = tweet['coordinates'] #no 'coordinates' found in tweets

    #if place exists in tweet, return it
    if place:
        #print(place)
        return tweet['place']['country']


    #if no 'place' exists in tweet, we try to get location info from user's location
    user_loc = tweet['user']['location']  # get user's location
    # print(user_loc)

    #if user's location is empty, return None
    if user_loc == '':
        return None

    #try to find some 'standar' keywords refering to specific countries (geograpy could not identify them)
    country = recognizeSpecificCountries(user_loc)
    #print(country)

    if country:
        return country


    #find country using geograpy
    places = geograpy.get_place_context(text=user_loc)
    #print(places)
    #print(places.countries)

    if not places.countries:
        #print('No country found\n')
        return None

    #geograpy returns a list with all possible counries, we take the first one !!!!! should we?? should we not???
    return places.countries[0]



def update_tweet_location(tweet, location):
    """
    :param location: tweet's location to import in a new column in db
    :return: None
    """
    collection.update(
        {
            'id': tweet['id']
        },
        {
            "$set": {
                "country": location
            }
        }
    )



total_tweets = collection.estimated_document_count()
tweets_loc_found_count = 0
tweet_index = 0

tweets = collection.find().batch_size(20)

for tweet in tweets:
    tweet_country = getTweetCountry(tweet)
    update_tweet_location(tweet, tweet_country)

    tweet_index += 1
    if tweet_country:
        tweets_loc_found_count += 1

    print('[Country Added]', tweet['id'], '[' + str(tweet_index) + '/' + str(total_tweets) + ']')


print('Found country for ', tweets_loc_found_count, ' tweets from ', total_tweets, ' tweets (total)')
print('Percentage of tweets with country: ', tweets_loc_found_count/total_tweets *100)


