from mongodb_scripts.mongo_db import *
import timeit

start = timeit.default_timer()

coll = db["covid_dupls2"]
cursor = coll.aggregate(
    [
        {"$group": {"_id": "$id", "unique_ids": {"$addToSet": "$_id"}, "count": {"$sum": 1}}},
        {"$match": {"count": { "$gte": 2 }}}
    ],
    allowDiskUse=True
)


response = []
for doc in cursor:
    print(doc)
    del doc["unique_ids"][0]
    for id in doc["unique_ids"]:
        response.append(id)
        #print(id)

#print(response)
coll.delete_many({"_id": {"$in": response}})

stop = timeit.default_timer()
print('Time: %.5f' % (stop - start))
