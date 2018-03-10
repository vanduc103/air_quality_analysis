import pickle
import numpy
import math

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge, Reshape
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint


def split_features(X):
    X = numpy.array(X)
    X_list = []

    stations = X[..., [0]]
    X_list.append(stations)

    year = X[..., [1]]
    X_list.append(year)

    month = X[..., [2]]
    X_list.append(month)

    day_of_week = X[..., [3]]
    X_list.append(day_of_week)

    hour = X[..., [4]]
    X_list.append(hour)

    season = X[..., [5]]
    X_list.append(season)

    return X_list

class Model(object):

    def __init__(self, train_ratio):
        self.train_ratio = train_ratio
        self.__load_data()

    def evaluate(self):
        if self.train_ratio == 1:
            return 0
        total_sqe = 0
        num_real_test = 0
        for record, aqi in zip(self.X_val, self.y_val):
            guessed_aqi = self.guess(record)
            #sqe = ((aqi - guessed_aqi) / aqi) ** 2
            sqe = ((aqi - guessed_aqi)) ** 2
            total_sqe += sqe
            num_real_test += 1
        result = math.sqrt(total_sqe / num_real_test)
        return result

    def __load_data(self):
        f = open('feature_train_data.pickle', 'rb')
        (self.X, self.y) = pickle.load(f)
        self.X = numpy.array(self.X)
        self.y = numpy.array(self.y)
        self.num_records = len(self.X)
        self.train_size = int(self.train_ratio * self.num_records)
        self.test_size = self.num_records - self.train_size
        self.X, self.X_val = self.X[:self.train_size], self.X[self.train_size:]
        self.y, self.y_val = self.y[:self.train_size], self.y[self.train_size:]


class NN_with_EntityEmbedding(Model):

    def __init__(self, train_ratio):
        super().__init__(train_ratio)
        self.build_preprocessor(self.X)
        self.nb_epoch = 20
        self.checkpointer = ModelCheckpoint(filepath="best_model_weights.hdf5", verbose=1, save_best_only=True)
        self.max_log_y = numpy.max(numpy.log(self.y))
        self.min_log_y = numpy.min(numpy.log(self.y))
        self.__build_keras_model()
        self.fit()

    def build_preprocessor(self, X):
        X_list = split_features(X)

    def preprocessing(self, X):
        X_list = split_features(X)
        return X_list

    def __build_keras_model(self):
        models = []

        model_stations = Sequential()
        model_stations.add(Embedding(33, 10, input_length=1))
        model_stations.add(Reshape(target_shape=(10,)))
        models.append(model_stations)

        model_year = Sequential()
        model_year.add(Embedding(4, 2, input_length=1))
        model_year.add(Reshape(target_shape=(2,)))
        models.append(model_year)

        model_month = Sequential()
        model_month.add(Embedding(12, 6, input_length=1))
        model_month.add(Reshape(target_shape=(6,)))
        models.append(model_month)

        model_dow = Sequential()
        model_dow.add(Embedding(7, 3, input_length=1))
        model_dow.add(Reshape(target_shape=(3,)))
        models.append(model_dow)

        model_hour = Sequential()
        model_hour.add(Embedding(24, 10, input_length=1))
        model_hour.add(Reshape(target_shape=(10,)))
        models.append(model_hour)

        model_season = Sequential()
        model_season.add(Embedding(4, 2, input_length=1))
        model_season.add(Reshape(target_shape=(2,)))
        models.append(model_season)

        self.model = Sequential()
        self.model.add(Merge(models, mode='concat'))
        self.model.add(Dropout(0.02))
        self.model.add(Dense(1000, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(500, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(100, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dense(50, init='uniform'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def _val_for_fit(self, val):
        val = numpy.log(val) / self.max_log_y
        return val

    def _val_for_pred(self, val):
        val = numpy.exp(val * self.max_log_y)
        return val

    def fit(self):
        if self.train_ratio < 1:
            self.model.fit(self.preprocessing(self.X), self._val_for_fit(self.y),
                           validation_data=(self.preprocessing(self.X_val), self._val_for_fit(self.y_val)),
                           nb_epoch=self.nb_epoch, batch_size=128,
                           callbacks=[self.checkpointer],
                           )
            self.model.save('models.h5')
            # self.model.load_weights('best_model_weights.hdf5')
            print("Result on validation data: ", self.evaluate())
        else:
            self.model.fit(self.preprocessing(self.X), self._val_for_fit(self.y),
                           nb_epoch=self.nb_epoch, batch_size=128)

    def guess(self, feature):
        feature = numpy.array(feature).reshape(1, -1)
        return self._val_for_pred(self.model.predict(self.preprocessing(feature)))[0][0]
