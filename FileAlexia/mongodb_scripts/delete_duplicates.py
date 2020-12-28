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


duplicate_found = False
response = []

for duplicate in cursor:
    duplicate_found = True
    #print(doc)
    del duplicate["unique_ids"][0]
    for id in duplicate["unique_ids"]:
        response.append(id)
        #print(id)

#print(response)
coll.delete_many({"_id": {"$in": response}})

if not duplicate_found:
    print('No duplicates found')

stop = timeit.default_timer()
print('Time: %.5f' % (stop - start))
