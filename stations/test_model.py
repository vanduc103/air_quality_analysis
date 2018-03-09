import pickle
from models import NN_with_EntityEmbedding
import math
import numpy
import sys
from keras.models import load_model
sys.setrecursionlimit(10000)

num_networks = 1
train_ratio = 0.95

with open('feature_train_data.pickle', 'rb') as f:
    X, y = pickle.load(f)
    num_records = len(y)

models = []
for i in range(num_networks):
    print("Fitting NN_with_EntityEmbedding...")
    models.append(NN_with_EntityEmbedding(train_ratio))

#with open('models.pickle', 'wb') as f:
#    pickle.dump(models, f, -1)


def evaluate_models(models, num_records):
    model0 = models[0]
    train_size = train_ratio * num_records
    total_sqe = 0
    num_real_test = 0
    if model0.train_ratio == 1:
        return 0
    for i in range(model0.train_size, num_records):
        record = X[i]
        aqi = y[i]
        if aqi == 0:
            continue
        guessed_aqi = numpy.mean([model.guess(record) for model in models])
        sqe = ((aqi - guessed_aqi)) ** 2
        total_sqe += sqe
        num_real_test += 1
        if num_real_test % 1000 == 0:
            print("{}/{}".format(num_real_test, num_records - model0.train_size))
            print(aqi, guessed_aqi)
    result = math.sqrt(total_sqe / num_real_test)
    return result

print("Evaluate combined models...")
r = evaluate_models(models, num_records)
print(r)

print("Testing...")
num_real_test = 0
total_sqe = 0
with open('feature_test_data.pickle', 'rb') as f:
    test_X, test_y = pickle.load(f)
    for i in range(len(test_X)):
        record = test_X[i]
        aqi = test_y[i]
        if aqi == 0:
            continue
        guessed_aqi = numpy.mean([model.guess(record) for model in models])
        sqe = ((aqi - guessed_aqi)) ** 2
        total_sqe += sqe
        num_real_test += 1
        if num_real_test % 1000 == 0:
            print("{}/{}".format(num_real_test, len(test_X)))
            print(aqi, guessed_aqi)
    error = math.sqrt(total_sqe / num_real_test)
    print(error)

