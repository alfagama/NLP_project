from FileAlexia.mongodb_scripts.mongo_db import *
import timeit


######################################
start = timeit.default_timer()
######################################

collection = db['covidtemp']


cursor = collection.aggregate([#{"$limit": 1000}, #in case we want to limit to the first N records
                                {"$group":{"_id":"$location", "count":{"$sum":1}}},
                                {"$sort":{"count":-1}}],
                                allowDiskUse=True)


#create dictionary, in case we need it
countries_dict = {}
for doc in cursor:
    print(doc)
    countries_dict.update({doc['_id']:doc['count']})

#print(countries_dict)

######################################
stop = timeit.default_timer()
print('Time: %.5f' % (stop - start))
######################################