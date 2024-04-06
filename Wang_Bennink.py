import math
import pandas as pd
from functools import reduce
from scipy import sparse
from scipy import optimize
import pennylane as qml
from pennylane import numpy as np
import sys
from sklearn.model_selection import train_test_split
np.set_printoptions(threshold=sys.maxsize)

#This is the unitary for endocing the data in the binary encoded data approach. See paper for reference
def createU_k(k_index: int, x_k: float):
    Z = qml.PauliZ(0)

    k_projector = np.array([0] * 2 ** QPU_len)
    k_projector[k_index] = 1
    k_projector = qml.Projector(k_projector, wires=[x for x in range(1, QPU_len + 1)])
    return qml.exp(Z @ k_projector, -1j * x_k)

#This is the unitary for regression coefficients.
def createU_m(phi_m: float, m_index: int):
    global N_L
    global N_M
    Z = qml.PauliZ(0)
    I = qml.Identity(wires=[x for x in range(1, N_L + 1)])

    projector = np.array([0] * (2 ** N_M))
    projector[m_index] = 1
    m_projector = qml.Projector(projector, wires=[x for x in range(1 + QPU_len - N_M, QPU_len + 1)])

    qml.exp(Z @ I @ m_projector, 1j * phi_m)

#The measurement operator
def createM_hat():
    global N_M
    global N_L

    I = sparse.csc_matrix(np.array([[1, 0], [0, 1]]))
    X = sparse.csc_matrix(np.array([[0, 1], [1, 0]]))

    N_L_sum = [I] * N_L
    N_M_sum = [I + X] * N_M

    N_L_tensor = reduce(lambda operator, product: sparse.kron(operator, product), N_L_sum)
    N_M_tensor = reduce(lambda operator, product: sparse.kron(operator, product), N_M_sum)
    M_hat = sparse.kron(N_L_tensor, N_M_tensor)
    M_hat = qml.SparseHamiltonian(M_hat.tocsr(copy=True), wires=[x for x in range(1, QPU_len + 1)])


    return M_hat


   
df = pd.read_csv("./Admission_Predict.csv")

X = np.array(df.iloc[:,1:-1])
y = np.array(df.iloc[:,-1])

X, X_test, y, y_test = train_test_split(X, y, test_size=0.36, random_state=42)

l = len(X) #Rows
m = len(X[0]) + 1 #Columns (including label)
data = np.empty((l, m))

for i in range(l):
    data[i] = np.flip(np.append(X[i], y[i])) #Reverse the data order

numBatches = 1 #Number of batches
batchSize = int(np.ceil(l / numBatches))

data = [data[i:i + batchSize] for i in range(0, len(data), batchSize)]

l = len(data[0]) #Rows
m = len(data[0][0]) #Columns (including label)

N_M = int(np.ceil(np.log2(m))) #Binary length for column items
N_L = int(np.ceil(np.log2(l))) #Binary length for row items

for index, element in enumerate(data):

    #Standardize column-wise and normalize globally
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

    data[index] = dataPadded.flatten()

QPU_len = N_M + N_L
epoch = 0
iteration = 0

sys.setrecursionlimit(9000)
M_hat = createM_hat()

all_U_k = []
for i in range(len(data)):
    temp = []
    for j in range(len(data[i])):
        temp.append(createU_k(j, data[i][j]))
    all_U_k.append(temp)
    
#The algorithm
dev = qml.device("default.qubit", wires= 1 + QPU_len)
@qml.qnode(dev, diff_method="parameter-shift")
def qn(phi, U_k_D):

    for i in range(QPU_len + 1):
        qml.Hadamard(i)
    for U in U_k_D:
        qml.apply(U)

    qml.Hadamard(0)
    qml.measure(0, reset=True, postselect=1)
    qml.Hadamard(0)

    for i in range(2 ** N_M):
        phi_value = phi[i]
        createU_m(phi_value, i)

    qml.Hadamard(0)
    qml.measure(0, postselect=0)
    return qml.expval(M_hat)

#Function for the optimizer
def calc_expval(phi):
    global epoch
    global iteration
    print(phi)
    expval = qn(phi, all_U_k[iteration % numBatches])

    if iteration % numBatches == 0: 
        epoch += 1 
        print(f"Epoch: {epoch}")
    iteration += 1
    print(f"Iteration: {iteration}")

    expval /= math.pow(math.cos(phi[0]), 2)
    print(f"Expectation Value: {np.round(expval,8).real}")
    return expval

init = [np.pi / 2]  * (2 ** N_M - 1)
init.insert(0, 3 * np.pi / 4, ) #Initial parameters
bounds = [(-np.pi, np.pi)] * (2 ** N_M - 1)
bounds.insert(0, ( np.pi / 2, 3 * np.pi / 2))
bounds = tuple(bounds) #Bounds

res = optimize.minimize(fun = calc_expval, x0 = init, method = 'Nelder-Mead', options={'maxiter' : 600, 'disp': True}, bounds = bounds, tol=1e-6)
print(res.x)


