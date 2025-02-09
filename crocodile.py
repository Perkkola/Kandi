from sklearn.preprocessing import PolynomialFeatures   
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score   
from sklearn.tree import export_text, plot_tree, DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import math
import numpy as np
f = open("02-week-crocodile-high-to-low.txt", 'r')
number_array = f.readline().split(' ')

X = []
y = []
window_size = 10

for i in range(len(number_array) - window_size):
    x = []
    for j in range(window_size):
        x.append(int(number_array[i+j]))
    y.append(int(number_array[window_size + i]))
    X.append(x)

X = np.array([z for z in range(2, 20)]).reshape(-1, 1)
y = [0, 1, 2, 4, 6, 9, 12, 16, 20, 25, 30, 36, 42, 49, 56, 64, 72, 81]

lin_regr = LinearRegression(fit_intercept=False)
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(X)
lin_regr.fit(x_train_poly, y)

y_pred=lin_regr.predict(x_train_poly)
y_pred_rounded = list(map(lambda x: round(x), y_pred))


accuracy = accuracy_score(y, y_pred_rounded)
print(accuracy)

z = list(zip(lin_regr.coef_, poly.get_feature_names_out()))

poly = list(map(lambda x: f'{x[0]}*{x[1]}', z))
string = "+".join(poly)
print(string)

# clf = DecisionTreeClassifier(random_state=0, max_depth=15)
# clf.fit(X, y)
# # y_pred = clf.predict(X)
# print(clf.score(X, y))

# mlp = MLPClassifier(random_state=1, max_iter=1000, hidden_layer_sizes=(1000,)).fit(X, y)
# print(mlp.score(X, y))
# # # print(mlp.coefs_)
# # print(mlp.n_layers_)
# # print(mlp.n_outputs_)
# # print(mlp.out_activation_)