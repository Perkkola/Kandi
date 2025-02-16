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

X=[[0.267333482606368, 0.07146719092244927, 0.019105573041392574], [-0.3392389344956481, 0.11508305467774264, -0.039040652847381827], [0.3846188560332424, 0.14793166441632005, 0.05689730753889853], [-0.4119751209870932, 0.16972350031233008, -0.06992185957552514], [-0.49126090234565334, 0.24133727417346554, -0.11855956708009702], [-0.47023834318324953, 0.22112409939972755, -0.10398103013961607], 
[0.47027534138540794, 0.22115889671516198, 0.10400557565314297], [-0.7877547702393104, 0.6205575780347887, -0.4888471923050579], [-0.6109322531268262, 0.37323821791062045, -0.22802326542117668], [0.6274396723045168, 0.39368054238159944, 0.2470107905045752], [0.9061157432417117, 0.8210457401504797, 0.7439624710718932], [-0.2502720204812259, 0.06263608423575515, -0.015676059356714702], [-0.34820233834005276, 0.12124486842548057, -0.04221774669748437], [-0.6349608682545549, 0.4031753042145782, -0.2560005412228829], [-0.7348022784736865, 0.5399343884501211, -0.3967450188594455], [0.7388599867130499, 0.5459140799656083, 0.4033540698698561], [-0.2323772615935391, 0.05399919170571211, -0.01254818429683793], [-0.16011367017095202, 0.02563638737561241, -0.0041047360726335635], [0.7403514750173759, 0.5481203065604041, 0.40580167744897144], [-0.8113531092923847, 0.6582938679584204, -0.5341087765961748], [-0.005914109915885257, 3.497669609717233e-05, -2.0685602521319202e-07], [0.9730720802048494, 0.9468692732741928, 0.9213720534269728], [0.6713405635402978, 0.45069815225460463, 0.3025719515211772], [0.10727969700082429, 0.011508933388588669, 0.0012346748867304623], [-0.5154044805131981, 0.2656417785330796, -0.1369129628674439], [-0.42503028612761384, 0.1806507441257213, -0.07678203746492168], [0.8923928998391699, 0.7963650876833628, 0.7106705499284309], [0.5725801815696876, 0.3278480643263764, 0.18771930419926722], [-0.7355128525536865, 0.5409791562716609, -0.39789712240145586], [0.17348786161688157, 0.030098038128398247, 0.00522164427375918], [-0.2423764186154318, 0.05874632830084303, -0.014238724660364719], [-0.606876489610616, 0.3682990736421041, -0.22351204893876192]]
y = [0.26416057960396744, -0.3327694975737698, 0.3752058658461667, -0.4004199738837568, -0.47173804932937513, -0.4530987717372865, 0.4531317538563176, -0.708771188830907, -0.5736313304789722, 0.5870740139915479, 0.7871138438174013, -0.24766751412114646, -0.341208580002706, -0.593146010040965, -0.6704404666221297, 0.6734456096790993, -0.23029153688811635, -0.15943042387880618, 0.6745474232233392, -0.7252194777718434, -0.005914075439941348, 0.8266184636069777, 0.6220361914790442, 0.10707403623551798, -0.492886825183003, -0.4123483733896474, 0.7785756412830955, 0.5418025074481567, -0.670967517229354, 0.1726188963101769, -0.2400102587033261, -0.5703044906280459]

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

for i in range(1, 2):
    f = open(f"./sin/sin_3f-m3-cs-grad-{i}_angles.txt")
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