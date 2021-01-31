from FileAlexia.mongodb_scripts.mongo_db import *
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

for duplicate in cursor:
    duplicate_found = True
    print(duplicate)

if not duplicate_found:
    print('No duplicates found')


stop = timeit.default_timer()
print('Time: %.5f' % (stop - start))
