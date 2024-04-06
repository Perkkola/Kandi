import numpy as np
import pandas as pd
import math
from scipy import optimize
from sklearn.model_selection import train_test_split

df = pd.read_csv("./Admission_Predict.csv")

X = np.array(df.iloc[:,1:-1])
y = np.array(df.iloc[:,-1])
X, X_test, y, y_test = train_test_split(X, y, test_size=0.36, random_state=42)

ul = len(X_test) #Rows
um = len(X_test[0]) + 1 #Columns (including label)

N_M = int(np.ceil(np.log2(um))) #Binary length for column items
N_L = int(np.ceil(np.log2(ul))) #Binary length for row items

testData = np.empty((ul, um))

for i in range(ul):
    testData[i] = np.flip(np.append(X_test[i], y_test[i])) #Reverse the testData order
squareSum = 0
testData = testData.transpose()

# #Standardize column-wise and normalize globally
for i in range(um):
    testData[i] = testData[i] - np.mean(testData[i])
    testData[i] = testData[i] / np.std(testData[i])
    for j in range(ul):
        squareSum += np.square(testData[i][j])

testData = testData.transpose()
squareSum = np.sqrt(squareSum)
testData = testData / squareSum

testDataPadded = np.zeros((2**N_L, 2**N_M))
for i in range(ul):
    testDataPadded[i] = np.append(testData[i], [0] * (int(2**N_M - um)))

testData = testDataPadded

l = len(X) #Rows
m = len(X[0]) + 1 #Columns (including label)

data = np.empty((l, m))

for i in range(l):
    data[i] = np.flip(np.append(X[i], y[i])) #Reverse the data order

numBatches = 1
batchSize = int(np.ceil(l / numBatches))

data = [data[i:i + batchSize] for i in range(0, len(data), batchSize)]

l = len(data[0]) #Rows
m = len(data[0][0]) #Columns (including label)

N_M = int(np.ceil(np.log2(m))) #Binary length for column items
N_L = int(np.ceil(np.log2(l))) #Binary length for row items

for index, element in enumerate(data):
    # #Standardize column-wise and normalize globally

    squareSum = 0
    element = element.transpose()
    for i in range(m):
        element[i] = element[i] - np.mean(element[i])
        element[i] = element[i] / np.std(element[i])
        for value in element[i]:
            squareSum += np.square(value)

    element = element.transpose()
    squareSum = np.sqrt(squareSum)
    element = element / squareSum

    dataPadded = np.zeros((2**N_L, 2**N_M))
    for i in range(len(element)):
        dataPadded[i] = np.append(element[i], [0] * (int(2**N_M - m)))

    data[index] = dataPadded

epsilon = 10**-15
epoch = 0
iteration = 0

def mse(phi, partialData):
    mse = 0
    for i in range(2 ** N_L):
        mid = 0
        for j in range(2 ** N_M):
            mid += partialData[i][j] * math.cos(phi[j])
        mse += math.pow(mid, 2)
    return mse / math.pow(math.cos(phi[0]), 2)

def trainmse(phi):
    global iteration
    global epoch
    error = mse(phi, data[iteration % numBatches])
    if iteration % numBatches == 0: epoch += 1
    iteration += 1
    return error


init = [np.pi / 2]  * (2 ** N_M - 1)
init.insert(0, 3 * np.pi / 4, )
bounds = [(-np.pi, np.pi)] * (2 ** N_M - 1)
bounds.insert(0, ( np.pi / 2, 3 * np.pi / 2))


res = optimize.minimize(fun = trainmse, x0 = init, method = 'Nelder-Mead', options={'maxiter' : 100 * 6, 'disp': True}, bounds = bounds)
print(res.x)
print(res.message)
print(res.fun)

y_true = testData[:ul, [0]].flatten()
y_pred = []

W = res.x #Plug custom weights here

for i in range(ul):

    X = testData[i, 1:um]
    X_W = [a * (-math.cos(b) / math.cos(W[0]))  for a, b in zip(X, W[1:])]
    y_pred.append(sum(X_W))


print(1 - (((y_true - y_pred) ** 2).sum()) / (((y_true - y_true.mean()) ** 2).sum()))

