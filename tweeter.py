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
#   print(redirect_url)
webbrowser.open(redirect_url)
user_pin_input = input("What's the pin value ?")
#   user_pin_input
auth.get_access_token(user_pin_input)  # this gives the access keys for that
#                           particular user. These access keys dont change
#   print(auth.access_token, auth.access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)  # wait_on_rate_limit to not get banned! Wait for limit to end and
# then continue!
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
query = "#Εμβολιο"
df = pd.DataFrame()
for i, status in enumerate(tweepy.Cursor(api.search, q=query)
                                 .items(50)):
    # print(i, status.author.screen_name, status.text, "\n")
    new_row = pd.DataFrame({'Id': [i],
                            'User': [status.author.screen_name],
                            'Text': [status.text]})
    df = df.append(new_row)

print(df.head(50))

df.to_csv(r'Data\Εμβολιο.csv', index=False)