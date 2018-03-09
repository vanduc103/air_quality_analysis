import pickle
import time
import math
import pandas as pd
from sklearn import preprocessing

def read_data():
    df = pd.read_csv('../aqi_data/aqi_seoul/aqi_seoul_syn')

    return df.values

def degToCompass(angle):
    angle = int(angle)
    val=int((angle/22.5)+.5)
    arr=["N","NNE","NE","ENE","E","ESE", "SE", "SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return arr[(val % 16)]

def read_weather(le_wind, le_cond):
    df = pd.read_csv('../weather_data/seoul.csv')
    df['wind_dir_enc'] = le_wind.fit_transform(df.apply(lambda row: degToCompass(row['wind_angle']), axis = 1))
    df['cond_enc'] = le_cond.fit_transform(df['condition'])
    #print(len(le_cond.classes_))
    
    return df.values


def feature_list(aqi_rec, we_rec):
    year = int(aqi_rec[0]) - 2015
    season = int(aqi_rec[1])
    month = int(aqi_rec[2]) - 1
    day_of_week = int(aqi_rec[3])
    hour = int(aqi_rec[4])
    
    temp = float(we_rec[1])
    hum = float(we_rec[5])
    pres = float(we_rec[6])
    cond = int(we_rec[9])
    wind_spd = float(we_rec[3])
    wind_dir = int(we_rec[8])

    return [year,
            month,
            day_of_week,
            hour,
            season,
            temp,
            hum,
            pres,
            cond,
            wind_spd,
            wind_dir
            ]


aqi_data = read_data()
print(aqi_data.shape)
train_size = 26304 # 2015, 2016, 2017
test_size = len(aqi_data) - train_size

le_wind = preprocessing.LabelEncoder()
le_cond = preprocessing.LabelEncoder()
weather_data = read_weather(le_wind, le_cond)
print(weather_data[0])

train_data_X = []
train_data_y = []

for i in range(train_size-1):
    aqi_record = aqi_data[i]
    weather_record = weather_data[i]
    if float(aqi_record[5]) > 0:
        fl = feature_list(aqi_record, weather_record)
        train_data_X.append(fl)
        train_data_y.append(float(aqi_record[5]))
print("Number of train datapoints: ", len(train_data_y))

test_data_X = []
test_data_y = []
for i in range(test_size):
    aqi_record = aqi_data[train_size + i]
    weather_record = weather_data[train_size + i]
    if float(aqi_record[5]) > 0:
        fl = feature_list(aqi_record, weather_record)
        test_data_X.append(fl)
        test_data_y.append(float(aqi_record[5]))
print("Number of test datapoints: ", len(test_data_X))

print(min(train_data_y), max(train_data_y))


with open('feature_train_data.pickle', 'wb') as f:
    pickle.dump((train_data_X, train_data_y), f, -1)
    print(train_data_X[0], train_data_y[0])

with open('feature_test_data.pickle', 'wb') as f:
    pickle.dump((test_data_X, test_data_y), f, -1)
    print(test_data_X[0], test_data_y[0])

