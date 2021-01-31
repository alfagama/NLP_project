from pymongo import MongoClient
import timeit

#####################################################################
connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterCovidDB

start = timeit.default_timer()

######################################

collection = db['copiedCollection']

collection.update({}, {"$unset": {"id_str":1, "truncated":1, "display_text_range":1, "entities":1,
"metadata":1, "source":1, "in_reply_to_status_id":1, "in_reply_to_status_id_str":1,
"in_reply_to_user_id":1,"in_reply_to_user_id_str":1, "in_reply_to_screen_name":1,
"contributors":1,"is_quote_status":1,"retweeted":1, "lang":1, "possibly_sensitive":1,
"quoted_status_id":1, "quoted_status_id":1, "quoted_status_id_str":1,
"quoted_status":1, "extended_entities":1,
"user.name":1,
"user.screen_name":1,
"user.description":1,
"user.url":1,
"user.entities":1,
"user.protected":1,
"user.followers_count":1,
"user.friends_count":1,
"user.listed_count":1,
"user.utc_offset":1,
"user.time_zone":1,
"user.verified":1,
"user.statuses_count":1,
"user.contributors_enabled":1,
"user.is_translator":1,
"user.is_translation_enabled":1,
"user.profile_background_color":1,
"user.profile_background_image_url":1,
"user.profile_background_image_url_https":1,
"user.profile_background_tile":1,
"user.profile_image_url":1,
"user.profile_image_url_https":1,
"user.profile_banner_url":1,
"user.profile_link_color":1,
"user.profile_sidebar_border_color":1,
"user.profile_sidebar_fill_color":1,
"user.profile_text_color":1,
"user.profile_use_background_image":1,
"user.has_extended_profile":1,
"user.default_profile":1,
"user.default_profile_image":1,
"user.following":1,
"user.follow_request_sent":1,
"user.notifications":1,
"user.translator_type":1}}, multi=True);

stop = timeit.default_timer()
print('Time: %.5f' % (stop - start))