import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers 
import pandas as pd
import numpy as np
(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
x_test=x_test.astype("float32")/255.0
x_train=x_train.astype("float32")/255.0
#Flatten dta
x_train=x_train.reshape(-1,28*28)
x_test=x_test.reshape(-1,28*28)
#Build a dense neural network
model=keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128,activation="relu"),
    layers.Dense(64,activation="relu"),
    layers.Dense(10,activation="softmax")
])
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy"
,metrics=["accuracy"])
# Step 6: Train the model
history=model.fit(x_train,y_train,validation_split=0.1,epochs=10,batch_size=32)
# Step 7: Evaluate on test data
test_loss,test_acc=model.evaluate(x_test,y_test)
print(f"Test Accuracy: {test_acc:.4f}")
# Step 8: Inspect predictions
predictions = model.predict(x_test)
# Example: Show first 5 predictions
for i in range(5):
    pred_class = np.argmax(predictions[i])
    print(f"Image {i} - Predicted: {pred_class}, True: {y_test[i]}")
    plt.imshow(x_test[i].reshape(28,28), cmap="gray")
    plt.show()

# Step 9: Save the model for reuse
model.save("mnist_dense_model.h5")
print("Model saved as mnist_dense_model.h5")