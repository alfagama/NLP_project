from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from FileAlexia.mongodb_scripts.mongo_db import *
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation as LDA
import warnings
#from pyLDAvis import sklearn as sklearn_lda
from gensim.models import CoherenceModel



warnings.simplefilter("ignore", DeprecationWarning)


sns.set_style('whitegrid')
#% matplotlib inline  # Helper function


# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]

    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))

    plt.figure(2, figsize=(15, 15 / 1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()  # Initialise the count vectorizer with the English stop words


collection = db['covid']
#tweet_texts = collection.distinct("text_preprocessed")
cursor = collection.aggregate([{"$limit": 1000},
                               {"$group" : {"_id": "$id", "text" : {"$push": "$text_preprocessed"}}}])


alltext = []
for doc in cursor:
    #print(doc["text"])
    alltext.append(doc["text"])
txtlist = [item for items in alltext for item in items]


count_vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(txtlist)

# Visualise the 10 most common words
#plot_10_most_common_words(count_data, count_vectorizer)


# Tweak the two parameters below
number_topics = 10
number_words = 10

# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)

# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)



'''doc_topic = lda.transform(lda)

for n in range(doc_topic.shape[0]):
    topic_most_pr = doc_topic[n].argmax()
    print("doc: {} topic: {}\n".format(n,topic_most_pr))'''


# Compute Coherence Score
'''coherence_model_lda = CoherenceModel(model=lda, texts=txtlist, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)'''
