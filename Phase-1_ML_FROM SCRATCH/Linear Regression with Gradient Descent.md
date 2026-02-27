# Linear Regression with Gradient Descent 📈
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/5873f190-7f5c-4454-981f-0733654426ec" />
<img width="2000" height="999" alt="image" src="https://github.com/user-attachments/assets/a7a7d978-8604-4a47-9d5b-5dddd9716e6b" />

A from-scratch implementation of Linear Regression using Gradient Descent optimization algorithm.

---

## 📋 Overview

This files implements **Linear Regression** from scratch using only NumPy. It demonstrates:
- The mathematics behind linear regression
- How Gradient Descent works to find optimal parameters
---

## 🧠 Theory

### Linear Regression
Linear Regression models the relationship between variables by fitting a linear equation:
```
y = mx + b
```
Where:
- **y**: target/dependent variable
- **x**: feature/independent variable  
- **m**: slope (weight)
- **b**: intercept (bias)

For multiple features:
```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

### Gradient Descent
Gradient Descent is an optimization algorithm that iteratively finds the optimal parameters by minimizing the cost function:

1. **Cost Function** (Mean Squared Error):
   ```
   MSE = (1/n) * Σ(y_true - y_pred)²
   ```

2. **Gradient Calculation**:
   ```
   ∂MSE/∂w = (-2/n) * Σx(y_true - y_pred)
   ∂MSE/∂b = (-2/n) * Σ(y_true - y_pred)
   ```

3. **Parameter Update**:
   ```
   w = w - α * (∂MSE/∂w)
   b = b - α * (∂MSE/∂b)
   ```
   Where **α** is the learning rate

---

## 📁 Files

| File | Description |
|------|-------------|
| `linear_regression-with-gradient_descent.py` | Main implementation with visualization |
| `Linear Regression with Gradient Descent.md` | This documentation |

---




