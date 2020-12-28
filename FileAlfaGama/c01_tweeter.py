import tweepy
import webbrowser
import pandas as pd

#####################################################################
#   API key & Secret
#####################################################################
comsumer_key = 'pTGjQtJMUiv1RyCbOY6YD7a53'
consumer_secret = 'xOhpCUMraZfGOEG21dAg3nFBWoSTgD4Xp0EvaiM2uA5aaIjYRJ'

#####################################################################
#   Connecting
#####################################################################
callback_uri = 'oob'  # typically https://..././../..
auth = tweepy.OAuthHandler(comsumer_key, consumer_secret, callback_uri)
redirect_url = auth.get_authorization_url()
webbrowser.open(redirect_url)
user_pin_input = input("What's the pin value ?")
auth.get_access_token(user_pin_input)  # this gives the access keys for that
#                           particular user. These access keys dont change
api = tweepy.API(auth, wait_on_rate_limit=True)  # wait_on_rate_limit to not
# get banned! Wait for limit to end and then continue!
me = api.me()
print(me.screen_name)

#####################################################################
#   Fetch timeline
#####################################################################
#   print(api.home_timeline())
#   print(len(api.home_timeline()))
#   print(len(api.home_timeline(count=50)))

#####################################################################
#   Print timeline - Readable, not Reliable!
#####################################################################
# my_timeline = api.home_timeline(count=50)
# for status in my_timeline:
#     print(status.text, "\n")
# # although it returns all asked, this method above is not always reliable, so..

#####################################################################
#    Print timeline - Method with Cursor. Better!
#####################################################################
# for i, status in enumerate(tweepy.Cursor(api.home_timeline, count=50)
#                                  .items(50)):  # or more.. but..limit!
#     print(i, status.text, "\n")

#####################################################################
#   Tweets from specific user
#####################################################################
# print("--------------------------User-------------------------------")
# other_user = "CyberpunkGame"
# for i, status in enumerate(tweepy
#                            .Cursor(
#                                api.user_timeline,
#                                screen_name=other_user)
#                            .items(20)):
#     print(i, status.text)

#####################################################################
#   Stuff for specific user
#####################################################################
# user = api.get_user(other_user)
# print(dir(user))

#####################################################################
#   Friends of user
#####################################################################
# user_friends = []
# for i, _id in enumerate(tweepy
#                         .Cursor(api.friends_ids, screen_name=other_user)
#                         .items(30)):
#     #print(i, _id)
#     print(i, api.get_user(_id).screen_name)
#     user_friends.append(_id)

# print(api.get_user(user_friends[0]).screen_name)

#####################################################################
#   Query search
#####################################################################
query = "#coronavirus"
df = pd.DataFrame()
for i, status in enumerate(tweepy.Cursor(api.search,
                                         q=query)
                                 .items(500)):
    new_row = pd.DataFrame({'Num': [i],
                            'Id': [status.id],
                            'Created': [status.created_at],
                            'User': [status.author.screen_name],
                            'Text': [status.text],
                            'Likes': [status.favorite_count],
                            'Retweets': [status.retweet_count],
                            'Geolocation': [status.geo],
                            'Location': [status.user.location],
                            'Language': [status.lang],
                            'User_Id': [status.user.id],
                            'User_Name': [status.user.name],
                            'User_Followers': [status.user.followers_count],
                            'User_Following': [status.user.friends_count]
                            })
    df = df.append(new_row)

print(df.head(100))

df.to_csv(r'Data\coronavirus_20201213.csv', index=False)