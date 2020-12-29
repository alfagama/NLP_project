import tweepy
import sys
from FileAlexia.credentials.credentials import *


auth = tweepy.AppAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True,
                 wait_on_rate_limit_notify=True)

if not api:
    print("Can't Authenticate")
    sys.exit(-1)
