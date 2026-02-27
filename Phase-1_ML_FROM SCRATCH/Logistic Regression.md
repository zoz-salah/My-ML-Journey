## 🧠 Logistic Regression From Scratch (Using NumPy)

## 📌 file Overview

This file implements **Binary Logistic Regression from scratch** using only **NumPy** — without using any machine learning libraries 

The goal of this implementation is to deeply understand:

* How Logistic Regression works mathematically
* How the Sigmoid function converts linear output into probabilities
* How Binary Cross-Entropy relates to gradient updates
* How Gradient Descent optimizes model parameters

---

## 📊 Problem Type

Binary Classification

Example:

| Input (x) | Output (y) |
|-----------|------------|
| 1         | 0          |
| 2         | 0          |
| 3         | 1          |
| 4         | 1          |

The model learns to predict whether a sample belongs to class **0 or 1**.

---

## 🧮 Background

### 1️⃣ Linear Equation

```
z = w * x + b
```

### 2️⃣ Sigmoid Function

```
σ(z) = 1 / (1 + e^(-z))
```

This converts any real number into a value between **0 and 1** ( statics probability).

### 3️⃣ Prediction Rule

```
ŷ = 1 if σ(z) ≥ 0.5, else 0
```

### 4️⃣ Gradients Used

```
dw = (1/n) * Σ (ŷ - y) * x
db = (1/n) * Σ (ŷ - y)
```


## 🚀 Code Structure

```python

---
