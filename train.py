import matplotlib.pylab as plt
import datetime as dt

from Scripts.process import *

stocks = ['AAPL5Y.csv', 'GOOGL5Y.csv', 'IBM5Y.csv', 'MSFT5Y.csv', 'SAP5Y.csv']
i = 0

while i < 5:

    TRAIN_SIZE = 30
    TARGET_TIME = 1
    LAG_SIZE = 1
    EMB_SIZE = 1

    timeseries = []
    dates = []

    print('Data loading...')
    timeseries, dates = load_close(stocks[i])
    dates = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dates]

    plt.plot(dates, timeseries)

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

plt.show()