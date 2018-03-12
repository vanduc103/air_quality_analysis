import pickle
import time
import math
import pandas as pd

def get_time(offset):
    offset = int(offset)
    t = time.strptime("17 06 01", "%y %m %d")
    t = time.mktime(t)
    t = t + offset * 3600

    return t

def read_data():
    df = pd.read_csv('daegu_data.csv')
    df['time'] = df.apply(lambda row: get_time(row['timerange']), axis = 1)
	
    return df.values


def feature_list(record):
    t = float(record[7])
    struct = time.localtime(t)

    mapidx = int(record[1])
    year = struct[0] - 2017
    month = struct[1] - 1
    day_of_week = struct[6]
    hour = struct[3]

    return [mapidx,
            year,
            month,
            day_of_week,
            hour
            ]


alldata = read_data()
print(alldata.shape)
print(alldata[0])

train_ratio = 0.97
train_size = int(len(alldata) * train_ratio)
train_data = alldata[:train_size]
test_data = alldata[train_size+1:]

train_data_X = []
train_data_y = []

for record in train_data:
    y_val = float(record[5])
    if y_val > 0:
        fl = feature_list(record)
        train_data_X.append(fl)
        train_data_y.append(y_val)
print("Number of train datapoints: ", len(train_data_y))

test_data_X = []
test_data_y = []
for record in test_data:
    y_val = float(record[5])
    if y_val > 0:
        fl = feature_list(record)
        test_data_X.append(fl)
        test_data_y.append(y_val)
print("Number of test datapoints: ", len(test_data_X))

print(min(train_data_y), max(train_data_y))


with open('feature_train_so2.pickle', 'wb') as f:
    pickle.dump((train_data_X, train_data_y), f, -1)
    print(train_data_X[0], train_data_y[0])

with open('feature_test_so2.pickle', 'wb') as f:
    pickle.dump((test_data_X, test_data_y), f, -1)
    print(test_data_X[0], test_data_y[0])

