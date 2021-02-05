## MSc Data & Web Science, Aristotle University of Thessaloniki (AUTH)
### Course: Text Mining and Natural Language Processing
#### Project: *“Sentiment Analysis of tweets about Covid-19 based on geolocation”*
----------------------------------------------------
**Team Members**:
1. Georgios Arampatzis
2. Alexia Fytili
3. Eleni Tsiolaki

----------------------------------------------------

## [PROJECT REPORT PDF](https://drive.google.com/file/d/1TqKVmbV1fPO6cM1QEBT-2u31Rmjo70dg/view?usp=sharing)

----------------------------------------------------

## Dataset(s):
Utilizing the Twitter API, we aim to gather data based on a variety of hashtags related to Coronavirus used on Twitter.

In more detail the 3 datasets with their respective entries number:  

| Tables        | Entries       |
| ------------- |:-------------:|
| Covid         | 429.019       |
| Quarantine    | 364.075       | 
| Vaccine       | 64.200        | 

## Methodology:
- Unsupervised Methods : Vader & Textblob
- Supervised Methods : LSTM, BiLSTM & CNN Neural Networks

## Evaluation measure:
- Unsupervised ML on data retrieved from Twitter
- Sentiment Analysis

----------------------------------------------------

```
.
└── twitter_sentiment_analysis_based_on_geolocation
    ├── country_visualization
    │   ├── country_visualization.py
    │   ├── country_visualization_CreateMap.py
    │   └── country_visualization_createDFs.py
    ├── credentials
    │   └── credentials.py
    ├── data_analysis
    │   ├── create_wordcloud.py
    │   ├── group_by_country.py
    │   ├── group_by_date.py
    │   ├── ngrams.py
    │   └── topic_modeling.py
    ├── mongodb_useful
    │   ├── copydb.py
    │   ├── deleteColumns.py
    │   ├── delete_duplicates.py
    │   ├── find_duplicates.py
    │   ├── groupDates.py
    │   ├── mongo_db.py
    │   └── move_docs_to_another_collection.py
    ├── outputs
    │   ├── Dates
    │   │   ├── Date-Tweet_plot_vaccine.png
    │   │   └── vaccine_dates.txt
    │   ├── Locations
    │   │   └── vaccines_map.png
    │   ├── Ngrams
    │   │   ├── bigrams_vaccine__final.txt
    │   │   ├── fourgrams_vaccine_final.txt
    │   │   └── trigrams_vaccine_final.txt
    │   ├── TopicModeling
    │   │   └── vaccine_topics.txt
    │   ├── WordClouds
    │   │   ├── first_review.png
    |   │   └── vaccine_wordcloud.png
    ├── preprocessing
    │   ├── dictionaries.py
    │   ├── preprocessing.py
    │   └── tweet_location.py
    ├── supervised
    │   ├── CNN
    │   │   └── CNN.py
    │   ├── LSTM
    │   │   ├── plots
    |   |   |   ├── LSTM_accuracy.png
    |   |   |   └── LSTM_loss.png
    │   │   ├── BiLSTM.py
    │   │   ├── BiLSTM_glove.py
    │   │   ├── LSTM.py
    │   │   ├── LSTM_glove.py
    │   │   └── LoadModelAndPredict.py
    ├── supervised_that_we_never_ran
    │   ├── albert.py
    │   └── bert.py
    ├── unsupervised
    │   ├── Vader-TextBlob.py
    │   ├── unsupervised_Results.py
    │   └── unsupervised_lexicon_based.py
    ├── README.md
    ├── api.py
    └── tweet_collection.py
```
