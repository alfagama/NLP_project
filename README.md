## MSc Data & Web Science, Aristotle University of Thessaloniki (AUTH)
### Course: Text Mining and Natural Language Processing
#### Project: *“Analysis of public reactions to COVID-19 related tweets based on geolocation”*
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
    ├── Dates
    │   ├── .png
    │   ├── .png
    │   ├── Date-Tweet_plot_vaccine.png
    │   ├── .txt
    │   ├── .txt
    │   └── vaccine_dates.txt
    ├── Locations
    │   ├── .png
    │   ├── .png
    │   └── vaccines_map.png
    ├── Ngrams
    │   ├── .txt
    │   ├── .txt
    │   ├── bigrams_vaccine__final.txt
    │   ├── .txt
    │   ├── .txt
    │   ├── fourgrams_vaccine_final.txt
    │   ├── .txt
    │   ├── .txt
    │   └── trigrams_vaccine_final.txt
    ├── TopicModelling
    │   ├── .txt 
    │   ├── .txt
    │   └── vaccine_topics.txt
    ├── WordClouds
    │   ├── .png
    │   ├── .png
    │   └── vaccine_wordcloud.png
    ├── Useful_for_mongodb
    │   ├── duplicates/etc.withdb.py
    │   ├── .py
    │   └── .py
    ├── README.md
    ├── country_visualization.py
    ├── dictionaries.py
    ├── ngrams.py
    ├── preprocessing.py
    ├── unsupervised_lexicon_based.py
    ├── .py
    ├── .py
    ├── .py
    ├── .py
    ├── .py
    └── .py
```
