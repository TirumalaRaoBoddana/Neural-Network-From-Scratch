# Neural Network From Scratch

This repository contains a **Python implementation of a feedforward neural network built from scratch**. It demonstrates the core principles of neural networks, including **forward propagation, backpropagation, gradient descent**, and **loss computation**, all without high-level libraries like TensorFlow or PyTorch.  

---

## 🏗️ Architecture Overview

We use a simple feedforward neural network:

- **Input layer:** 2 neurons  
- **Hidden layer:** 2 neurons with ReLU activation  
- **Output layer:** 1 neuron with Sigmoid (binary classification) or Linear (regression) activation  

### Network Diagram

![Neural Network Diagram](./assets/nn_diagram.png)  
*Every weight, bias, pre-activation, and activation is annotated for clarity.*

---

## 📐 Mathematical Formulation

### Forward Propagation

Hidden layer neurons:

\[
z_1^{(1)} = w_{11}^{(1)} x_1 + w_{21}^{(1)} x_2 + b_1^{(1)}, \quad a_1^{(1)} = \text{ReLU}(z_1^{(1)})
\]

\[
z_2^{(1)} = w_{12}^{(1)} x_1 + w_{22}^{(1)} x_2 + b_2^{(1)}, \quad a_2^{(1)} = \text{ReLU}(z_2^{(1)})
\]

Output neuron:

\[
z^{(2)} = w_{11}^{(2)} a_1^{(1)} + w_{21}^{(2)} a_2^{(1)} + b^{(2)}, \quad \hat{y} = \sigma(z^{(2)})
\]

Where:

\[
\text{ReLU}(z) = \max(0, z), \quad \sigma(z) = \frac{1}{1 + e^{-z}}
\]

---

### Loss Functions

- **Binary Cross-Entropy (classification):**
\[
\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \big[y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)\big]
\]

- **Mean Squared Error (regression):**
\[
\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

---

### Backpropagation

Output layer error:

\[
\delta^{(2)} = \hat{y} - y
\]

Hidden layer errors:

\[
\delta^{(1)} = (\delta^{(2)} W^{(2)T}) \odot \mathbb{I}(Z^{(1)} > 0)
\]

Weight updates:

\[
W^{(l)} \leftarrow W^{(l)} - \eta \frac{1}{n} A^{(l-1)T} \delta^{(l)}, \quad
b^{(l)} \leftarrow b^{(l)} - \eta \frac{1}{n} \sum \delta^{(l)}
\]

---

## ⚡ Features

- Fully connected neural network from scratch  
- Forward propagation with **ReLU/Sigmoid activations**  
- Manual **backpropagation** with gradient descent  
- Supports **regression and binary classification**  
- Example **Jupyter notebook** for testing  

---

## 🛠️ Installation

```bash
git clone https://github.com/TirumalaRaoBoddana/Neural-Network-From-Scratch.git
cd Neural-Network-From-Scratch
pip install -r requirements.txt
