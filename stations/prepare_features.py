import pickle
import time
import math
import pandas as pd

def read_data():
    df = pd.read_csv('../../aqi_data/aqi_seoul/aqi_seoul_stations_syn')

    return df.values


def feature_list(record):
    stationidx = int(record[0])
    year = int(record[1]) - 2015
    season = int(record[2])
    month = int(record[3]) - 1
    day_of_week = int(record[4])
    hour = int(record[5])

    return [stationidx,
            year,
            month,
            day_of_week,
            hour,
            season
            ]


alldata = read_data()
print(alldata.shape)
train_ratio = 0.97
train_size = int(len(alldata) * train_ratio)
train_data = alldata[:train_size]
test_data = alldata[train_size+1:]

train_data_X = []
train_data_y = []

for record in train_data:
    if float(record[6]) > 0:
        fl = feature_list(record)
        train_data_X.append(fl)
        train_data_y.append(float(record[6]))
print("Number of train datapoints: ", len(train_data_y))

test_data_X = []
test_data_y = []
for record in test_data:
    if float(record[6]) > 0:
        fl = feature_list(record)
        test_data_X.append(fl)
        test_data_y.append(float(record[6]))
print("Number of test datapoints: ", len(test_data_X))

print(min(train_data_y), max(train_data_y))


with open('feature_train_data.pickle', 'wb') as f:
    pickle.dump((train_data_X, train_data_y), f, -1)
    print(train_data_X[0], train_data_y[0])

with open('feature_test_data.pickle', 'wb') as f:
    pickle.dump((test_data_X, test_data_y), f, -1)
