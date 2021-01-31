#   https://plotly.com/python/choropleth-maps/
import plotly
import plotly.express as px
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pymongo import MongoClient
from collections import Counter


def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

#Connection to MongoDB
#####################################################################
connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterCovidDB

collections = db.collection_names()

#####################################################################

#Get all countries
all_countries = []
i = 0
errors = 0
for collection in collections:
    tweets = db[collection].find().batch_size(10)
    if collection == 'copiedCollection':
        for tweet in tweets:
            try:
                location = tweet["location"]
                all_countries.append(str(location))
            except KeyError:
                errors+= 1
                print('missing field: ', tweet["id"])
            i+=1
            print("Count documents ", i)

print('Errors: ', errors)

all_countries = remove_values_from_list(all_countries, 'None')
unique_tweets_frequency = Counter(all_countries)
print(unique_tweets_frequency)

#Creating the Country-Counts DataFrame
df = []
dict_Countries_Count = {}
for index, value in enumerate(Counter(all_countries)):
    df.append([value, Counter(all_countries)[value]])
    dict_Countries_Count[value] = Counter(all_countries)[value]
df_countries_counts = pd.DataFrame(df, columns=['country', 'counts'])

print(df_countries_counts)

df_countries_counts.to_csv("quarantine_Country_vis.csv")