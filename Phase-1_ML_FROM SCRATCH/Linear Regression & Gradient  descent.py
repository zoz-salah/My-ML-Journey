import numpy as np


x = np.array([1,2,3,4])
y = np.array([50,60,70,80])

w = 0
b = 0

learning_rate = 0.01
n = len(x)

#gradient descent for decreasing eror
#repeat untill convergence
for epoch in range(1000):
    
    y_pred = w * x + b
    
    dw = (-2/n) * np.sum(x * (y - y_pred))  #= −n2​∑x(y−y^​)
    db = (-2/n) * np.sum(y - y_pred)     #−n2​∑(y−y^​)
    
    w -= learning_rate * dw
    b -= learning_rate * db

print("w: ", w)
print("b: ", b)
