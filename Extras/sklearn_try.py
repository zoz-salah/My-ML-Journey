from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Polynomial_features
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

print(model.predict([[6]]))
print(model.coef_)
print(model.intercept_)
print(model.predict([[5, 6]]))
print(model.coef_)
