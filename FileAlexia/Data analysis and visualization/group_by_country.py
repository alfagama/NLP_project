from mongo_db import *


collection = db['covidtemp']


countries_qty_cursor = collection.aggregate([
    {"$group":{"_id":"$country", "count":{"$sum":1}}},
    {"$sort":{"count":-1}}],
    allowDiskUse=True)


#create dictionary, in case we need it
countries_qty_dict = {}
for pair in countries_qty_cursor:
    print(pair)
    countries_qty_dict.update({pair['_id']:pair['count']})

#print(countries_qty_dict)

