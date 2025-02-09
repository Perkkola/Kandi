import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd




classical_adam = np.array([2.00260554, 1.54799347, 1.42548578, 1.52242757, 1.56077938, 1.53826396,
 1.4993929 , 1.47214224])
hardware = np.array([2.62935018,1.44960612,1.39585109,1.51584347,1.51942441,1.51269392
,1.46387139,1.51859484])
noiseless = np.array([2.93143268,1.40415329,1.37117185,1.43289854,1.40703251,1.42792815
,1.44377302,1.35058973])

sum_cl_adam = 0
sum_hardware = 0
sum_noiseless = 0

for i in range(len(classical_adam)):

    cl_adam_cos = math.cos(classical_adam[i])
    hardware_cos = math.cos(hardware[i])
    noiseless_cos = math.cos(noiseless[i])

    classical_adam[i] = cl_adam_cos
    hardware[i] = hardware_cos
    noiseless[i] = noiseless_cos

    sum_cl_adam += cl_adam_cos
    sum_hardware += hardware_cos
    sum_noiseless += noiseless_cos

sum_cl_adam = np.sqrt(sum_cl_adam)
sum_hardware = np.sqrt(sum_hardware)
sum_noiseless = np.sqrt(sum_noiseless)

classical_adam * sum_cl_adam
hardware * sum_hardware
noiseless * sum_noiseless


classical_adam = classical_adam * np.std(classical_adam)
hardware = hardware * np.std(hardware)
noiseless = noiseless * np.std(noiseless)

classical_adam = classical_adam + np.mean(classical_adam)
hardware = hardware + np.mean(hardware)
noiseless = noiseless + np.mean(noiseless)


df = pd.read_csv("./Admission_Predict.csv")

X = np.array(df.iloc[:,1:-1])
y = np.array(df.iloc[:,-1])
X, X_test, y, y_test = train_test_split(X, y, test_size=0.36, random_state=42)

reg = LinearRegression().fit(X, y)

print(reg.score(X, y))

sklearn_coef = reg.coef_

classical_adam = classical_adam[1:]
hardware = hardware[1:]
noiseless = noiseless[1:]

# print(f"Classical adam: {classical_adam}")
# print(f"Hardware: {hardware}")
# print(f"Noiseless: {noiseless}")
# print(f"Sklearn: {reg.coef_}")

for i in range(len(classical_adam)):
    hardware[i] = abs(math.log2(classical_adam[i]) - math.log2(hardware[i])) 
    noiseless[i] = abs(math.log2(classical_adam[i]) - math.log2(noiseless[i])) 
    # classical_adam[i] = abs(sklearn_coef[i] - classical_adam[i])

print(f"Hardware deviation: {hardware}")
print(f"Noiselessdeviation: {noiseless}")
# print(f"Classical adam deviation: {classical_adam}")