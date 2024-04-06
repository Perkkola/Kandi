import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("./Admission_Predict.csv")

X = np.array(df.iloc[:,1:-1])
y = np.array(df.iloc[:,-1])
X, X_test, y, y_test = train_test_split(X, y, test_size=0.36, random_state=42)
reg = LinearRegression().fit(X, y)
print(reg.score(X_test, y_test))
