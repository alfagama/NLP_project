from pymongo import MongoClient
from datetime import datetime
from collections import Counter
import json

#####################################################################
connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterDB

collections = db.collection_names()

#####################################################################
all_dates = []
count = 0
for collection in collections:
    tweets = db[collection].find().batch_size(10)
    if collection == 'vaccine_test3':
        for tweet in tweets:
            # date = db[collection].find_one({'created_at': tweet["created_at"]})["created_at"]
            # new_datetime = datetime.strftime(datetime.strptime(date, '%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d')
            # all_dates.append(new_datetime)
            # print(new_datetime)
            date = db[collection].find_one({'tweet_date': tweet["tweet_date"]})["tweet_date"]
            all_dates.append(date)
            count += 1
            print(count)

#####################################################################
# unique_dates = Counter(all_dates).keys()
# print("No of unique items in the list are:", len(unique_dates))

unique_dates_frequency = Counter(all_dates)
print(unique_dates_frequency.items())
#####################################################################
with open('Dates/dates.txt', 'w') as file:
    for item in unique_dates_frequency.items():
        file.write(json.dumps(item))

#####################################################################
