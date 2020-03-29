import pandas as pd
from Custom.Database import MysqlDB as Database
from sklearn.model_selection import train_test_split


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


dfhotelFilteredNeg.rename(
    {'Negative_Review' : 'review', 'label' : 'label'}, axis = 1, inplace=True
)
dfhotelFilteredPos.rename(
    {'Positive_Review' : 'review', 'label' : 'label'}, axis = 1, inplace=True
)


model_testPos = dfhotelFilteredPos.sample(15000, random_state=0)
model_testNeg = dfhotelFilteredNeg.sample(15000, random_state=0)

final_modelPos = dfhotelFilteredPos.drop(model_testPos.index)
final_modelNeg = dfhotelFilteredNeg.drop(model_testNeg.index)

trainPos = final_modelPos.sample(100000, random_state=0)
trainNeg = final_modelNeg.sample(100000, random_state=0)

testPos = final_modelPos.drop(trainPos.index)
testNeg = final_modelNeg.drop(trainNeg.index)

model_tests = pd.concat([model_testPos, model_testNeg], axis = 0)
train = pd.concat([trainPos, trainNeg], axis = 0)
test = pd.concat([testPos, testNeg], axis = 0)

model_tests['set'] = 'model_tests' # for comparing models
train['set'] = 'train' # for training final models
test['set'] = 'test' # for testing final models


allCleanedData = pd.concat([model_tests, train, test], axis = 0)


db = Database()
db.to_database(allCleanedData, 'clean_data')

