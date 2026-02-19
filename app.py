import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
(x_train,y_train),(x_test,y_test)=keras.datasets.cifar10.load_data()
x_train=x_train/255.0
x_test=x_test/255.0
class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]
model=keras.Sequential([
    layers.Flatten(input_shape=(32,32,3)),
    layers.Dense(256,activation="relu"),
    layers.Dense(10,activation="softmax")

])
model.compile(optimizer='adam',loss="sparse_categorical_crossentropy",metrics=['accuracy'])
model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))
test_loss,test_acc=model.evaluate(x_test,y_test)
print("Test Accurracy:",test_acc)
index=10
img=x_test[index]
true_label=y_test[index][0]
prediction=model.predict(np.array([img]))
predicted_label=np.argmax(prediction)
plt.imshow(img)
plt.title(f"Predicted:{class_names[predicted_label]},Actual:{class_names[true_label]}")
plt.axis('off')
plt.show()