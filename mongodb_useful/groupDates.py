from pymongo import MongoClient
from collections import Counter
import matplotlib.pyplot as plt

def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

#####################################################################
connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterCovidDB

collections = db.collection_names()

#####################################################################

all_dates = []
for collection in collections:
    tweets = db[collection].find().batch_size(20).limit(300000)
    if collection == 'copiedCollection':
        for tweet in tweets:
            date = tweet["tweet_date"]
            all_dates.append(str(date))

#all_dates = remove_values_from_list(all_dates, 'None')
unique_tweets_frequency = Counter(all_dates)
print(unique_tweets_frequency)

dict_Dates = {}
for index, value in enumerate(Counter(all_dates)):
    dict_Dates[value] = Counter(all_dates)[value]
print(dict_Dates)

dates = list(dict_Dates.keys())
tweets = list(dict_Dates.values())
print(dates)
print(tweets)
# Setting axes limits
# plt.ylim(-1.2, 1.2)
# plt.xlim(0, 10)
plt.bar(dates, tweets)
plt.xticks(rotation=90)
plt.rcParams["font.size"] = 10
plt.title("Tweet frequency by Date")
plt.savefig("Date-Tweet_plot.png", dpi=100)
plt.show()

