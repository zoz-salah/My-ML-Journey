# Neural Network from Scratch with NumPy 🧠

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)

<img width="791" height="388" alt="image" src="https://github.com/user-attachments/assets/a795ca64-700f-47dd-9d5b-cd7ade396508" />


A complete from scratch implementation of a Neural Network for classification problems, built layer by layer using only NumPy.

---

## 📋 Overview

This implementation demonstrates how to build a fully-connected (dense) neural network from the ground up. You'll understand exactly what happens inside each layer during forward propagation and backpropagation.

---

## 🏗️ Neural Network Architecture

### Network Structure

```
INPUT LAYER        HIDDEN LAYERS          OUTPUT LAYER
    [x₁] ──────┐   ┌─[h₁]─┐   ┌─[h₁]─┐   ┌─[y₁]
    [x₂] ──────┼──►│  h₂  │──►│  h₂  │──►│  y₂  │
    [x₃] ──────┘   └─[h₃]─┘   └─[h₃]─┘   └─[y₃]
    [x₄] ──────┐     Layer 1     Layer 2    Output
    [x₅] ──────┘                              (Softmax)
    
Features: n_in   Hidden 1: n_h1   Hidden 2: n_h2   Classes: n_out
```

### Layer-by-Layer Breakdown

#### 1. **Input Layer** (Features)
- **Role**: Receives raw data
- **Size**: Number of features (e.g., 784 for MNIST, 4 for Iris dataset)
- **No computation** - just passes data forward
- **Shape**: (batch_size, n_features)

#### 2. **Hidden Layers**
- **Role**: Learn complex patterns and representations
- **Components**:
  - Linear transformation: `Z = W·X + b`
  - Activation function: `A = activation(Z)`
- **Multiple hidden layers** allow learning hierarchical features

#### 3. **Output Layer** (Predictions)
- **Role**: Produce final classification
- **Size**: Number of classes (e.g., 10 for MNIST digits)
- **Activation**: Softmax (for multi-class) or Sigmoid (for binary)

---


## 🔄 Activation Functions Explained

| Function | Formula | Output Range | Best For | Layer Type |
|----------|---------|--------------|----------|------------|
| **ReLU** | `max(0, x)` | [0, ∞) | Hidden layers | Prevents vanishing gradient |
| **Sigmoid** | `1/(1+e⁻ˣ)` | (0, 1) | Binary output | Probability output |
| **Tanh** | `(eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)` | (-1, 1) | Hidden layers | Zero-centered |
| **Softmax** | `eˣ/Σeˣ` | (0, 1) | Multi-class output | Probability distribution |


<img width="1200" height="603" alt="image" src="https://github.com/user-attachments/assets/2b206bf7-1822-45a2-9986-4e38cea7317c" />
<img width="1200" height="603" alt="image" src="https://github.com/user-attachments/assets/2672ee2c-055b-48d7-9afb-eb59a9529716" />

---

