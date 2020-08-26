#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

dataset = tf.keras.datasets.fashion_mnist.load_data()
(x_train, y_train), (x_test, y_test) = dataset

# Visualising 25 samples from the training set
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fig = plt.Figure(figsize=(15, 15))
for i in range(25):
    r = random.randint(0, 59999)
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(classes[y_train[r]])
    plt.imshow(x_train[r], cmap='Greys')
plt.tight_layout()
plt.show()

# Reshaping to 4D arrays and normalisation for the CNN
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train/255.0
x_test = x_test/255.0

#%%
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense 

model = tf.keras.models.Sequential()

# Add layers
model.add(Conv2D(filters=5, kernel_size=(3,3), input_shape=(28,28,1), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=5, kernel_size=(3,3), input_shape=(28,28,1), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

# Compilation of layers and fitting
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])
model.fit(x=x_train, y=y_train, epochs = 5)

#%%
model.evaluate(x_test, y_test)