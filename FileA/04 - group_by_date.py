from pymongo import MongoClient
from datetime import datetime
from collections import Counter

connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterDB

collections = db.collection_names()

all_dates = []
for collection in collections:
    tweets = db[collection].find().batch_size(10)
    if collection == 'vaccine':
        for tweet in tweets:
            date = db[collection].find_one({'created_at': tweet["created_at"]})["created_at"]
            new_datetime = datetime.strftime(datetime.strptime(date, '%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d')
            all_dates.append(new_datetime)
            print(new_datetime)

# unique_dates = Counter(all_dates).keys()
# print("No of unique items in the list are:", len(unique_dates))

unique_dates_frequency = Counter(all_dates)
print(unique_dates_frequency.items())
