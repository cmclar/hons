import numpy as np
import matplotlib.pylab as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, classification_report
import datetime as dt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D
from keras.callbacks import Callback
from keras.layers import Embedding
from process import *

stocks = ['AAPL5Y.csv', 'GOOGL5Y.csv', 'IBM5Y.csv', 'MSFT5Y.csv', 'SAP5Y.csv']
model_number = 0

while model_number < 3:
    i = 0
    while i < 5:
        timeseries = []
        dates = []

        print('Data loading...')
        timeseries, dates = load_close(stocks[i])
        dates = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dates]

        TRAIN_SIZE = 20
        TARGET_TIME = 1
        LAG_SIZE = 1
        EMB_SIZE = 1

        X, Y = split(timeseries, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False, scale=True)
        X, Y = np.array(X), np.array(Y)
        X_train, X_test, Y_train, Y_test = training_data(X, Y, 0.9)

        Xp, Yp = split(timeseries, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False, scale=False)
        Xp, Yp = np.array(Xp), np.array(Yp)
        X_trainp, X_testp, Y_trainp, Y_testp = training_data(Xp, Yp, 0.9)

        i = i+1

        if model_number == 0:
            print('Building model...')
            model = Sequential()
            model.add(Dense(500, input_shape=(TRAIN_SIZE,)))
            model.add(Activation('relu'))
            model.add(Dropout(0.25))
            model.add(Dense(250))
            model.add(Activation('relu'))
            model.add(Dense(1))
            model.add(Activation('linear'))
            model.compile(optimizer='adam',
                          loss='mse')

            model.fit(X_train,
                      Y_train,
                      nb_epoch=5,
                      batch_size=128,
                      verbose=1,
                      validation_split=0.1)
            score = model.evaluate(X_test, Y_test, batch_size=128)
            print(score)
            model.save('feedModel.h5')

        elif model_number == 1:
            model = Sequential()
            model.add(Convolution1D(input_shape=(TRAIN_SIZE, EMB_SIZE),
                                    activation="relu", filters=64, kernel_size=2, strides=1, padding="valid"))
            model.add(MaxPooling1D(pool_size=2))

            model.add(Convolution1D(input_shape=(TRAIN_SIZE, EMB_SIZE),
                                    activation="relu", filters=64, kernel_size=2, strides=1, padding="valid"))
            model.add(MaxPooling1D(pool_size=2))

            model.add(Dropout(0.25))
            model.add(Flatten())

            model.add(Dense(250))
            model.add(Dropout(0.25))
            model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('linear'))
            model.compile(optimizer='adam',
                          loss='mse')

            X_train = np.expand_dims(X_train, axis=2)
            Y_train = np.expand_dims(Y_train, axis=2)
            X_test = np.expand_dims(X_test, axis=2)
            Y_test = np.expand_dims(Y_test, axis=2)

            model.fit(X_train,
                      Y_train,
                      nb_epoch=5,
                      batch_size=128,
                      verbose=1,
                      validation_split=0.1)
            score = model.evaluate(X_test, Y_test, batch_size=128)
            print(score)
            model.save('convModel.h5')

        elif model_number == 2:
            model = Sequential()
            model.add(LSTM(32, return_sequences=True,
                           input_shape=(TRAIN_SIZE,EMB_SIZE)))  # returns a sequence of vectors of dimension 32
            model.add(LSTM(32, return_sequences=True,
                           input_shape=(TRAIN_SIZE, EMB_SIZE)))  # returns a sequence of vectors of dimension 32
            model.add(LSTM(32))  # return a single vector of dimension 32
            model.add(Dense(1))
            model.add(Activation('linear'))

            model.compile(optimizer='adam', loss='mse')

            X_train = np.expand_dims(X_train, axis=2)
            Y_train = np.expand_dims(Y_train, axis=2)
            X_test = np.expand_dims(X_test, axis=2)
            Y_test = np.expand_dims(Y_test, axis=2)

            model.fit(X_train,
                      Y_train,
                      nb_epoch=20,
                      batch_size=128,
                      verbose=1,
                      validation_split=0.1)
            score = model.evaluate(X_test, Y_test, batch_size=128)
            print("Score:")
            print(score)
            model.save('lstmModel.h5')

        #params = []
        #for xt in X_testp:
        #    xt = np.array(xt)
        #    mean_ = xt.mean()
        #    scale_ = xt.std()
        #    params.append([mean_, scale_])

        #steps_ahead = 7
        #curr_steps = 0
        #new_predicted = X_test

        #while curr_steps < steps_ahead:
        #    predicted = model.predict(new_predicted)
        #    new_predicted = np.expand_dims(predicted, axis=2)

        #predicted = model.predict(X_test, steps=2)
        #new_predicted = np.expand_dims(predicted, axis=2)

        #for pred, par in zip(predicted, params):
        #    a = pred * par[1]
        #    a += par[0]
        #    new_predicted.append(a)

        #mse = mean_squared_error(predicted, new_predicted)
        #print(mse)

        #try:
        #    fig = plt.figure()
        #    plt.plot(Y_test[:150], color='black')  # BLUE - trained RESULT
        #    plt.plot(predicted[:150], color='blue')  # RED - trained PREDICTION
        #    plt.plot(Y_testp[:150], color='green')  # GREEN - actual RESULT
        #    plt.plot(new_predicted[:150], color='red')  # ORANGE - restored PREDICTION

        #except Exception as e:
        #    print
        #    str(e)
    #plt.show()
    model_number = model_number + 1
