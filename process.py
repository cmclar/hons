import numpy as np
from sklearn import preprocessing


def load_delta(file):
    stockFile = open(file, 'rb').readlines()[1:]
    data = []
    dates = []
    for line in stockFile:
        try:
            openPrice = float(line.split(b',')[1])
            closePrice = float(line.split(b',')[4])
            data.append(closePrice - openPrice)
            dates.append(str(line.split(b',')[0]))

        except:
            continue

    return data[::-1], dates[::-1]


def load_close(file):
    stockFile = open(file, 'rb').readlines()[1:]
    data = []
    dates = []
    for line in stockFile:
        try:
            closePrice = float(line.split(b',')[4])
            data.append(closePrice)
            closeDate = (line.split(b',')[0])
            dates.append(closeDate.decode())

        except:
            continue


    return data, dates


def split(data, train, predict, step, binary, scale):
    X = []
    Y = []
    for i in range(0,len(data), step):
        try:
            xi = data[i:i+train]
            yi = data[i+train+predict]

            if binary:
                if yi > 0.:
                    yi = [1., 0.]
                else:
                    yi = [0., 1.]

                if scale:
                    xi = preprocessing.scale(xi)

            else:
                dataset = np.array(data[i:i+train+predict])

                if scale:
                    dataset = preprocessing.scale(dataset)

                xi = dataset[:-1]
                yi = dataset[-1]

        except:
            break

        X.append(xi)
        Y.append(yi)

    return X, Y


def shuffle(a, b):
    assert len(a) == len(b)
    shuffledA = np.empty(a.shape, dtype = a.dtype)
    shuffledB = np.empty(b.shape, dtype = b.dtype)

    permutation = np.random.permutation(len(a))
    for oldIndex, newIndex in enumerate(permutation):
        shuffledA[newIndex] = a[oldIndex]
        shuffledB[newIndex] = b[oldIndex]

    return shuffledA, shuffledB


def training_data(x, y, percentage):
    xTrain = x[0:int(len(x) * percentage)]
    yTrain = y[0:int(len(y) * percentage)]

    xTrain, yTrain = shuffle(xTrain, yTrain)

    xTest = x[int(len(x) * percentage):]
    yTest = y[int(len(y) * percentage):]

    return xTrain, xTest, yTrain, yTest


#stocks = ['AAPL5Y.csv', 'GOOGL5Y.csv', 'IBM5Y.csv', 'MSFT5Y.csv', 'SAP5Y.csv']

#i = 0
#while i < 5:
#    x, y = load_close(stocks[i])
#    print(x)
#    i = i+1
