import numpy as np
import matplotlib.pyplot as plt

# Default values 
# adam = Adam
#     learning_rate=0.001,  # α - step size
#     beta1=0.9,            
#     beta2=0.999,          
#     epsilon=1e-8          



def f(x):
    return x**2     #for power 

def df(x):
    return 2*x           #same

class SGD:
    def __init__(self, lr=0.1):  
        self.lr = lr
    def update(self, x, grad):
        return x - self.lr * grad

class Adam:
    def __init__(self, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):  
        self.lr = lr  # 0.1 or 0.001
        self.beta1 = beta1 #0.9
        self.beta2 = beta2 #0.999
        self.eps = eps
        self.m = 0
        self.v = 0
        self.t = 0
    
    def update(self, x, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


x_sgd = 5.0     #sigmoid gradient dessent
x_adam = 5.0    #adam

#comparing
sgd = SGD(lr=0.1)
adam = Adam(lr=0.1)

sgd_history = [x_sgd]
adam_history = [x_adam]

for i in range(50):

    grad_sgd = df(x_sgd)
    x_sgd = sgd.update(x_sgd, grad_sgd)
    sgd_history.append(x_sgd)
    
    grad_adam = df(x_adam)
    x_adam = adam.update(x_adam, grad_adam)
    adam_history.append(x_adam)

#visual compare (by AI) using matplotlib
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(sgd_history, 'b-', label='SGD', linewidth=2)
plt.plot(adam_history, 'r-', label='Adam', linewidth=2)
plt.xlabel('Step')
plt.ylabel('x value')
plt.title('Optimizer Comparison: x → 0')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(np.array(sgd_history)**2, 'b-', label='SGD', linewidth=2)
plt.plot(np.array(adam_history)**2, 'r-', label='Adam', linewidth=2)
plt.xlabel('Step')
plt.ylabel('Loss (x²)')
plt.title('Loss Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
