import pandas as pd

dataset = pd.read_csv("Data/Εμβολιο_20201210.csv",
                      sep=',',
                      header=None)
pd.set_option('display.max_columns', None)

print(dataset.head(10))

###################################################
#   Pre-Processing
###################################################
#
# 1. We removed the hashtag symbol and its content (e.g., #COVID19), @users, and URLs from
# the messages because the hashtag symbols or the URLs did not contribute to the message
# analysis.
# 2. We removed all non-English characters (non-ASCII characters) because the study focused
# on the analysis of messages in English.
# 3. We removed repeated words. For example, sooooo terrified was converted to so terrified.
# 4. We removed special characters, punctuations, and numbers from the dataset as they did
# not help with detecting the profanity comments.
#
# Additional Ideas:
# 1. After merging all .csv to 1 (based on issue) delete duplicate tweets / retweets / etc..
#
