from pymongo import MongoClient
import pandas as pd

def findLabelPosition(label):
    if label == 'Positive':
        x = 0
    elif label == 'Negative':
        x = 1
    else:
        x = 2
    return x

def createFinalDF(dataframe):
    pos = []
    neg = []
    neut = []

    for i, j in dataframe.iterrows():
        print(j[1])
        # printing the third element of the column
        i = 0
        for x in j[1]:
            if i == 0:
                pos.append(x)
            elif i == 1:
                neg.append(x)
            else:
                neut.append(x)
            i += 1
    countriesList = dataframe["Country"].tolist()
    return(pd.DataFrame(list(zip(countriesList, pos, neg, neut)),
                            columns=['Country', 'Positive', 'Negative', 'Neutral']))


#####################################################################
connection = MongoClient("mongodb://localhost:27017/")
db = connection.TwitterCovidDB

collections = db.collection_names()

#####################################################################
dictVader = {}
dictTextblob = {}
docCount = 0
errors = 0
for collection in collections:
    tweets = db[collection].find().batch_size(10)
    if collection == 'copiedCollection':
        for tweet in tweets:

            location = tweet["location"]
            vader = tweet["Vader"]
            textblob = tweet["TextBlob"]

            dictVader.setdefault(location, [0,0,0])
            dictTextblob.setdefault(location, [0,0,0])

            x_Vader = findLabelPosition(vader)
            x_TextBlob = findLabelPosition(textblob)

            dictVader[location][x_Vader] += 1
            dictTextblob[location][x_TextBlob] += 1

            docCount += 1
            print("Count documents ", docCount)


# Create Vader and TextBlob dataframes with their results per Country
#---------------------------------------------------------------------

vaderDF = pd.DataFrame(dictVader.items(), columns=['Country', 'Labels'])
textblobDF = pd.DataFrame(dictTextblob.items(), columns=['Country', 'Labels'])

vaderFinal = createFinalDF(vaderDF)
textblobFinal = createFinalDF(textblobDF)

#---------------------------------------------------------------------




# Save Vader and TextBlob results in CSV
#------------------------------------

vaderFinal.to_csv("quarantine_vader_results.csv")
textblobFinal.to_csv("quarantine_textblob_results.csv")

#------------------------------------
