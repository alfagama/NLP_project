# Import the wordcloud library
from wordcloud import WordCloud
from mongo_db import *
import matplotlib.pyplot as plt

collection = db['covidtemp']
tweet_texts = collection.distinct("text_preprocessed")
#print(tweet_texts)

#Should we delete keywords such as covid19?!!!!!!!!!!todo
# Join the different processed titles together.
long_string = ','.join(list(tweet_texts))

# Create a WordCloud object
wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')

# Generate a word cloud
wordcloud.generate(long_string)

# Save the word cloud
wordcloud.to_file("img/wordcloud.png")

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

