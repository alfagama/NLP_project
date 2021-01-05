## MSc Data & Web Science, Aristotle University of Thessaloniki (AUTH)
### Course: Text Mining and Natural Language Processing
#### Project: *“Analysis of public reactions to COVID-19 related tweets based on geolocation”*
----------------------------------------------------
**Team Members**:
1. Georgios Arampatzis
2. Alexia Fytili
3. Eleni Tsiolaki

----------------------------------------------------

## [PROJECT REPORT PDF](https://drive.google.com/file/d/1BWTRuGgoJlr7fDSwpupI3h6CBZsR4kt5/view?usp=sharing)
## [PROJECT REPORT DOC](https://docs.google.com/document/d/1D9RShWGWAr4y8_9XqsGT8kg2dIgVL8IE/edit)

----------------------------------------------------

## Dataset(s):
Utilizing the Twitter API, we aim to gather data based on a variety of hashtags related to Coronavirus used on Twitter.

In more detail the 3 datasets with their respective entries number:  

| Tables        | Entries       | Cols  |
| ------------- |:-------------:| -----:|
| Covid         | 1.002.599     |  122  |
| Quarantine    | 955.470       |  122  |
| Vaccine       | 136.088       |  122  |

## Methodology:
-
-

## Evaluation measure:
- Unsupervised ML on data retrieved from Twitter
- Sentiment Analysis

----------------------------------------------------

```
.
└── twitter_sentiment_analysis_based_on_geolocation
    ├── Dates
    │   ├── .txt
    │   ├── .txt
    │   └── vaccine_dates.txt vaccines_map.png
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
