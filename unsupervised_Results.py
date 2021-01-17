from pymongo import MongoClient
import pandas as pd

def findLabelPosition(label):
    if label == '1' or label == 1:
        x = 0
    elif label == '-1' or label == 0:
        x = 1
    elif label == '0':
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
dictBiLSTM = {}
docCount = 0
errors = 0
for collection in collections:
    tweets = db[collection].find().batch_size(10)
    if collection == 'copiedCollection':
        for tweet in tweets:

            location = tweet["location"]
            vader = tweet["vader_preprocessed_label"]
            textblob = tweet["textblob_preprocessed_label"]
            bilstm = tweet["BiLSTM_label"]

            dictVader.setdefault(location, [0,0,0])
            dictTextblob.setdefault(location, [0,0,0])
            dictBiLSTM.setdefault(location, [0,0,0])

            x_Vader = findLabelPosition(vader)
            x_TextBlob = findLabelPosition(textblob)
            x_bilstm = findLabelPosition(bilstm)

            dictVader[location][x_Vader] += 1
            dictTextblob[location][x_TextBlob] += 1
            dictBiLSTM[location][x_bilstm] += 1

            docCount += 1
            print("Count documents ", docCount)


# Create Vader and TextBlob dataframes with their results per Country
#---------------------------------------------------------------------

vaderDF = pd.DataFrame(dictVader.items(), columns=['Country', 'Labels'])
textblobDF = pd.DataFrame(dictTextblob.items(), columns=['Country', 'Labels'])
bilstmDF = pd.DataFrame(dictBiLSTM.items(), columns=['Country', 'Labels'])

vaderFinal = createFinalDF(vaderDF)
textblobFinal = createFinalDF(textblobDF)
bilstmFinal = createFinalDF(bilstmDF)

#---------------------------------------------------------------------




# Save Vader and TextBlob results in CSV
#------------------------------------

vaderFinal.to_csv("vader_results.csv")
textblobFinal.to_csv("textblob_results.csv")
bilstmFinal.to_csv("bilstm_results.csv")

#------------------------------------
