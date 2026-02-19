Here’s a clean, balanced **README.md** — not too long, not too short 👇

---

# 🖼️ CIFAR-10 Image Classification (TensorFlow)

This project builds a simple image classifier using **TensorFlow** and **Keras** to classify images from the CIFAR-10 dataset into 10 categories.

---
![TensorFlow Logo](tensor.png)

## 📦 Dataset

CIFAR-10 contains:

* 60,000 color images (32×32 pixels)
* 10 classes:

`Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck`

The dataset is loaded directly from Keras:

```python
keras.datasets.cifar10.load_data()
```

---

## 🧠 Model Architecture

A simple Dense Neural Network:

* Flatten layer (32×32×3 → 1D vector)
* Dense (256 neurons, ReLU)
* Dense (10 neurons, Softmax)

Compiled with:

* **Optimizer:** Adam
* **Loss:** Sparse Categorical Crossentropy
* **Metric:** Accuracy

---

## 🚀 Training & Evaluation

The model is trained for 10 epochs and evaluated on test data:

```python
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
model.evaluate(x_test, y_test)
```

Expected accuracy for this Dense-only model: **~45–55%**

---

## 🔍 Prediction Example

The script selects a test image, predicts its class, and displays:

* The image
* Predicted label
* Actual label

Using `matplotlib` for visualization.

---

## 🛠️ Requirements

```bash
pip install tensorflow numpy matplotlib
```

## 📌 Notes

This implementation uses a fully connected network.
For better performance on image classification, a CNN (Conv2D layers) is recommended.

---
