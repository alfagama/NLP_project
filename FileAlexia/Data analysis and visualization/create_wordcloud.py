from wordcloud import WordCloud
from FileAlexia.mongodb_scripts.mongo_db import *
import matplotlib.pyplot as plt
import timeit


######################################
start = timeit.default_timer()
######################################

collection = db['covidtemp2']

cursor = collection.aggregate([#{"$limit": 12000},
                               {"$group" : {"_id": "$id", "text" : {"$push": "$text_preprocessed"}}}])


alltext = []
for doc in cursor:
    #print(doc["text"])
    alltext.append(doc["text"])


# Convert list of list to list
txtlist = [item for items in alltext for item in items]

# Convert list to string
long_string = ''.join(txtlist)

# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')

# Generate a word cloud
wordcloud.generate(long_string)

# Save the word cloud
wordcloud.to_file("../img/wordcloud2.png")

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

######################################
stop = timeit.default_timer()
print('Time: %.5f' % (stop - start))
######################################







#2nd method, same result
###################################################################################
'''tweet_texts = collection.distinct("text_preprocessed")

# Join the different processed titles together.
long_string = ','.join(list(tweet_texts))

print(long_string)
# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')

# Generate a word cloud
wordcloud.generate(long_string)

# Save the word cloud
wordcloud.to_file("../img/wordcloud2.png")

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()'''