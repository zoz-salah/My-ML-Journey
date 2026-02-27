import numpy as np


x = np.array([1,2,3,4])
y = np.array([0,0,1,1])  

w = 0
b = 0

learning_rate = 0.1
n = len(x)
 #liniar regression normal
 

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


for epoch in range(1000):
    z = w * x + b
    y_pred = sigmoid(z)

    dw = (1/n) * np.sum((y_pred - y) * x)
    db = (1/n) * np.sum(y_pred - y)

    w -= learning_rate * dw
    b -= learning_rate * db

print("w:", w)
print("b:", b)
