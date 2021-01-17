from FileAlexia.mongodb_scripts.mongo_db import *
import gensim
import re
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim  # don't skip this


collection = db['covid']
NUM_TOPICS = 8


def main():

    # Get a list of all tweet texts from MongoDB
    #----------------------------------
    print('\nLoading from MongoDB..')
    cursor = collection.find({"tweet_date" : "2020-12-21"})

    data = []
    for doc in cursor:
        data.append(doc["text_preprocessed"])
        #print(doc["text_preprocessed"])

    #print(text_data)
    #----------------------------------


    # Create a dictionary
    # ----------------------------------
    print('\nCreating dictionary..')
    data = [d.split() for d in data]
    dictionary = gensim.corpora.Dictionary(data)
    #print(len(id2word))

    dictionary.filter_extremes(no_below=2, no_above=.99) # Filtering Extremes
    #print(len(id2word))
    # ----------------------------------


    # Creating a corpus object
    # ----------------------------------
    print('\nCreating corpus..')
    corpus = [dictionary.doc2bow(d) for d in data]
    # ----------------------------------


    # LDA model
    # ----------------------------------
    print('\nBuilding LDA model..')
    LDA_model = gensim.models.LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=5)  # Instantiating a Base LDA model
    # ----------------------------------


    # Create Topics
    # ----------------------------------
    print('\nTopics:')
    words = [re.findall(r'"([^"]*)"', t[1]) for t in LDA_model.print_topics()]   # Filtering for words
    topics = [' '.join(t[0:10]) for t in words]

    for id, t in enumerate(topics): # Getting the topics
        print(f"------ Topic {id} ------")
        print(t, end="\n\n")
    # ----------------------------------


    # Print topics with propabilities
    # ----------------------------------
    print('\nTopics with propabilities:')
    for i in LDA_model.print_topics():
        for j in i: print(j)
    # ----------------------------------


    # Get most frequent words of each topic
    # ----------------------------------
    print('\nMost frequent words by topic:')
    topic_words = []
    for i in range(NUM_TOPICS):
        tt = LDA_model.get_topic_terms(i, 20)
        topic_words.append([dictionary[pair[0]] for pair in tt])

    # output
    for i in range(NUM_TOPICS):
        print(f"------ Topic {id} ------")
        print(topic_words[i])
    # ----------------------------------


    # Compute Coherence and Perplexity
    # ----------------------------------
    #Compute Perplexity, a measure of how good the model is. lower the better
    print('\nComputing Coherence and Perplexity..')
    base_perplexity = LDA_model.log_perplexity(corpus)
    print('\nPerplexity: ', base_perplexity)

    # Compute Coherence Score
    coherence_model = CoherenceModel(model=LDA_model, texts=data,
                                   dictionary=dictionary, coherence='c_v')
    coherence_lda_model_base = coherence_model.get_coherence()
    print('\nCoherence Score: ', coherence_lda_model_base)
    # ----------------------------------


    # Creating Topic Distance Visualization
    # ----------------------------------
    print('\nCreating visualization..')
    visualisation = pyLDAvis.gensim.prepare(LDA_model, corpus, dictionary)
    pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')
    # ----------------------------------



if __name__ == "__main__":
    main()



