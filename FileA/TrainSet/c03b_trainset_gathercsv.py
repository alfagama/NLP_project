#####################################################################
import pandas as pd

#####################################################################
pd.set_option('display.max_columns', None)
cols = ["id", 'sentiment', 'tweet']
dataset_pos = pd.read_csv("train_pos.csv",
                          sep=',',
                          names=cols,
                          # header=None,  # no header, alternative header = header_col
                          # index_col=None,  # no index, alternative header = index_row
                          # skiprows=0  # how many rows to skip / not include in read_csv
                          )
print(dataset_pos.head())
print(dataset_pos.shape)

dataset_neg = pd.read_csv("train_neg.csv",
                          sep=',',
                          names=cols,
                          # header=None,  # no header, alternative header = header_col
                          # index_col=None,  # no index, alternative header = index_row
                          # skiprows=0  # how many rows to skip / not include in read_csv
                          )
print(dataset_neg.head())
print(dataset_neg.shape)
#####################################################################
dataset = pd.DataFrame()
dataset = dataset.append(dataset_pos)
dataset = dataset.append(dataset_neg)
print(dataset.shape)
dataset.reset_index(drop=True, inplace=True)
# dataset['tokens'], dataset['tokens_lemmatized'], dataset['full_text_lematized'] = \
#     dataset.apply(preprocessing(dataset))
# print(dataset.head())
dataset.to_csv(r'train_set.csv', index=False)
#
# # for row in dataset_pos:
# #     print(row['tweet'])
