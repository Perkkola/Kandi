import numpy as np
import pandas as pd
import math
from scipy import optimize
from sklearn.model_selection import train_test_split
from adam import Adam

df = pd.read_csv("./Admission_Predict.csv")

X = np.array(df.iloc[:,1:-1])
y = np.array(df.iloc[:,-1])
X, X_test, y, y_test = train_test_split(X, y, test_size=0.36, random_state=42)

X = np.array([[-1.60341029,-2.17465367,0.74887555],
[1.91216459,-0.43591851,0.33122774],
[-0.82526291,-2.27585162,0.1040535],
[0.35410691,-0.26710057,-0.1853427],
[0.71727858,-2.37481014,-2.50154728],
[-0.59997956,-0.75474662,0.16110463],
[0.0173233,-0.01229075,-0.1951486],
[-0.64781656,-0.57463777,0.87340028],
[-0.21907427,0.40714926,1.16389926],
[2.12276032,1.15836633,-0.28702311],
[0.71289885,-0.84483167,0.75987709],
[1.42682896,-1.3521705,0.52174947],
[0.2005193,2.55572106,-1.92078801],
[-0.29138296,0.71822356,-0.18879025],
[0.02960965,-0.25423061,1.00277756],
[-1.20171431,0.97095139,-1.35061643],
[1.45151341,-0.97263203,0.16887731],
[-1.1270828,-2.11558864,-0.74580644],
[-1.46428602,-0.44498525,-1.20583175],
[0.30243412,-1.58363399,2.49457863],
[0.78957546,-0.38739105,1.11763558],
[-0.67652503,0.28919442,0.63057408],
[1.15109342,0.00310858,0.36043627],
[0.32796474,-0.28123677,0.47578922],
[1.65793734,1.08060012,0.23154957],
[0.09325082,-0.06042793,-0.67259762],
[1.72121686,-0.26515998,0.68408241],
[-0.29348342,0.16532491,0.43069487],
[-0.12422754,0.09440476,-0.32619294],
[0.36258088,1.19141244,-0.66337902],
[-0.11168052,2.24801286,-0.35755231],
[1.73283522,-0.18962752,-2.12684498],
[0.3189123,-0.6094092,0.02119256],
[-0.18435388,-0.9261838,0.28053603],
[1.27867421,0.68507242,1.44917348],
[0.32102164,-0.59333144,0.64086836],
[0.55155558,1.66083182,-0.09950931],
[-0.84826867,0.28760851,-1.75347421],
[-1.07441945,-1.92848202,0.02073201],
[-0.6618314,-0.81113006,0.15707971],
[-0.85209116,0.71211576,-0.41645693],
[0.35006572,-0.01678624,-0.89681503],
[-0.44223831,-0.84981098,-1.78299192],
[0.04668856,0.84066495,-1.92336869],
[0.89933055,1.74514361,-0.728438],
[-1.24454805,-0.22616555,1.9299947],
[0.28371247,0.91806995,1.04868421],
[-0.42034452,0.01464703,0.93845679],
[0.72619616,1.81023182,-1.11106004],
[-1.11434391,-0.92873719,0.98259924],
[-1.54077716,-0.54068243,1.29326214],
[-0.68711576,-0.20317622,-0.75137156],
[-2.86120763,-0.56017894,1.46194975],
[-0.62540275,-0.47195563,-1.07510095],
[1.14653687,0.38471378,0.27692883],
[1.44495093,0.43648462,0.16876467],
[1.70572628,1.0602009,-0.65535834],
[-1.31212836,1.05303548,1.49791551],
[-0.90993618,0.03313295,1.09964943],
[-0.06458988,-0.04384766,0.67911689],
[0.37141956,0.88609774,1.14845481],
[-0.8626938,-0.66193366,1.46978516],
[-0.0826932,0.60270651,0.05997404],
[-0.93267417,2.36555639,-0.42678154]])

y= np.array([-142.77166873 , 170.26392479 , -73.48347645 ,  31.53056626  , 63.86828152
,  -53.42368298,    1.54251009 , -57.68320927  ,-19.50692197  ,189.01589635
 ,  63.47829893 , 127.04842454 ,  17.85474107 , -25.94546854  ,  2.63651728
 ,-107.00365202,  129.24639011, -100.35827589, -130.38369503 ,  26.92949174
  , 70.30577696,  -60.23948345 , 102.4962417  ,  29.202802  ,  147.62689364
 ,   8.30328668  ,153.26146092 , -26.1324989  , -11.06153099 ,  32.28511013
 ,  -9.9443134 ,  154.29598867 ,  28.39674972 , -16.4153316,   113.85635471
 ,  28.58457139 ,  49.11189015,  -75.53196698 , -95.66899874 , -58.93112567
 , -75.87233089 ,  31.17072814 , -39.37800721  ,  4.15726652 ,  80.07864519
 ,-110.81767581  , 25.26246906 , -37.4285287  ,  64.66232543 , -99.22397274
 ,-137.19465766 , -61.18250865, -254.76909436 , -55.68742705 , 102.09051468
 , 128.66205063 , 151.88214046 ,-116.83519631  ,-81.02299716   ,-5.75124481
  , 33.07212794 , -76.81641646  , -7.36320924 , -83.04764436])

# X = np.array([[-0.32741112, -0.11288069,  0.49650164],
#        [-0.94268847, -0.78149813, -0.49440176],
#        [ 0.68523899,  0.61829019, -1.32935529],
#        [-1.25647971, -0.14910498, -0.25044557],
#        [ 1.66252391, -0.78480779,  1.79644309],
#        [ 0.42989295,  0.45376306,  0.21658276],
#        [-0.61965493, -0.39914738, -0.33494265],
#        [-0.54552144,  1.85889336,  0.67628493]])

# y = np.array([ -8.02307406, -23.10019118,  16.79149797, -30.78951577,
#         40.73946101,  10.53434892, -15.18438779, -13.3677773 ])

# x = np.random.uniform(-1, 1, 2 ** 5)

# y = np.array([math.sin(z) for z in x])
# X = np.array([[z ** i for i in range(1, 4)] for z in x ])

X_test = X
y_test = y
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
weights = []

def mse(phi, partialData):
    mse = 0
    for i in range(2 ** N_L):
        mid = 0
        for j in range(2 ** N_M):
            mid += partialData[i][j] * math.cos(phi[j])
        mse += math.pow(mid, 2)
    return mse / math.pow(math.cos(phi[0]), 2)

def gradient(phi, partialData, index):
    mse = 0
    for i in range(2 ** N_L):
        mid = 0
        for j in range(2 ** N_M):
            mid += partialData[i][j] * math.cos(phi[j])
        mid *= 2
        mid *= -partialData[i][index]*math.sin(phi[index])
        mse += mid
    return mse / math.pow(math.cos(phi[0]), 2)

def trainmse(phi, grad = False):
    global iteration
    global epoch
    error = mse(phi, data[iteration % numBatches])
    if not grad:
        if iteration % numBatches == 0: epoch += 1
        iteration += 1
    return error

def gradient_descent(init, steps, lr):
        global iteration
        global epoch
        global weights
        x = init
        adam = Adam(n_iter=steps, lr=lr)
        adam.initialize_adam(x)

        for i in range(steps):
            weights.append(x)
            expval = trainmse(x, grad=True)
            grad = np.array([gradient(x, data[iteration % numBatches], j) for j in range(len(x))])
            x = adam.update_parameters_with_adam(x, grad, i)
            if iteration % numBatches == 0: epoch += 1
            iteration += 1

            print(x, expval)


init = [np.pi / 2]  * (2 ** N_M - 1)
init.insert(0, 3 * np.pi / 4, )
bounds = [(-np.pi, np.pi)] * (2 ** N_M - 1)
bounds.insert(0, ( np.pi / 2, 3 * np.pi / 2))


# gradient_descent(init, 50, 0.01)
# exit()

y_true = testData[:ul, [0]].flatten()
y_pred = []

all_weights = []

for i in range(1):
    f = open(f"./admission-multi-pass-nm/admission-m3-cs-grad-{i}_angles.txt")
    weights = []
    for j in range(100):
        line = f.readline()
        # line = line[1:-2]
        line = line.split(' ')
        line = [float(s) for s in line[:4]]
        weights.append(line)
    all_weights.append(weights)
# all_weights.append(weights)

all_accs = []
for weights in all_weights:
    accs = []
    for w in weights:
        y_pred = []
        for i in range(ul):

            X = testData[i, 1:um]
            X_W = [a * (-math.cos(b) / math.cos(w[0]))  for a, b in zip(X, w[1:])]
            y_pred.append(sum(X_W))

        accs.append(1 - (((y_true - y_pred) ** 2).sum()) / (((y_true - y_true.mean()) ** 2).sum()))
        # print(1 - (((y_true - y_pred) ** 2).sum()) / (((y_true - y_true.mean()) ** 2).sum()))
        # exit()
    all_accs.append(accs)
    # exit()
print(all_accs)
# print(np.array(all_accs).flatten())
exit()
mean_accs = []

for i in range(len(all_accs[0])):
    temp = []
    for j in range(len(all_accs)):
        temp.append(all_accs[j][i])
    mean_accs.append(np.array(temp).mean())

print(mean_accs)