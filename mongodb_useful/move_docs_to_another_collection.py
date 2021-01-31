from FileAlexia.mongodb_scripts.mongo_db import *
import timeit

start = timeit.default_timer()

from_coll = db["covid_trial2"]
to_coll = db["covid_trial1"]


tweet_index = 0
for tweet in from_coll.find().batch_size(20):
    tweet_index += 1
    if to_coll.count_documents({'id': tweet['id']}, limit=1) > 0:
        print('[Already exists]', tweet['id'], '[' + str(tweet['created_at']) + ']','[' + str(tweet_index) + '/' + str(from_coll.estimated_document_count()) + ']')
    else:
        to_coll.insert_one(tweet)
        print('[Added]', tweet['id'], '[' + str(tweet['created_at']) + ']','[' + str(tweet_index) + '/' + str(from_coll.estimated_document_count()) + ']')


stop = timeit.default_timer()
print('Time: %.5f' % (stop - start))
