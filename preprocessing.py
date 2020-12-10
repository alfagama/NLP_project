import pandas as pd

dataset = pd.read_csv("Data/Εμβολιο_20201210.csv",
                      sep=',',
                      header=None)
pd.set_option('display.max_columns', None)

print(dataset.head(10))

###################################################
#   Pre-Processing
###################################################
