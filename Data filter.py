import pandas as pd
from Custom.Database import MysqlDB as Database

dfhotel = pd.read_csv("Hotel_Reviews.csv")
dfhotel.head(5)

dfhotel.describe()

dfhotelFiltered = (
    dfhotel
    .loc[lambda df: df['Review_Total_Negative_Word_Counts'] >= 5]
    .loc[lambda df: df['Review_Total_Positive_Word_Counts'] >= 5]
)

dfhotelFilteredNeg = (
    dfhotelFiltered
    .loc[:, ['Negative_Review']]
)

dfhotelFilteredNeg['label'] = 0

dfhotelFilteredPos = (
    dfhotelFiltered
    .loc[:, ['Positive_Review']]
)

dfhotelFilteredPos['label'] = 1


dfhotelFilteredNeg.rename({'Negative_Review' : 'review', 'label' : 'label'}, axis = 1, inplace=True)
dfhotelFilteredPos.rename({'Positive_Review' : 'review', 'label' : 'label'}, axis = 1, inplace=True)

dfhotelCombi = pd.concat([dfhotelFilteredNeg, dfhotelFilteredPos], axis = 0)


db = Database()
db.to_database(dfhotelCombi, 'beef')

