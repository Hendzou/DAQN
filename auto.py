from datetime import datetime
from packaging import version
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from PIL import Image
from keras.preprocessing.image import img_to_array

n_inputs = 120
n_outputs = n_inputs
learning_rate = 0.01

filepath = r"C:\Users\Hend\Documents\DLCV\DLCV_1800809\Check23"
train_dir = r"C:\Users\Hend\Documents\DLCV\DLCV_1800809\Dataset"
X_train = []

if __name__ == '__main__':
    autoencoder = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(data_format="channels_last", input_shape=(120, 120, 3), filters=32, activation=tf.nn.relu,kernel_size=7),
    tf.keras.layers.MaxPool2D(pool_size=3),
    tf.keras.layers.Conv2D(filters=64, activation=tf.nn.relu,kernel_size=5),
    tf.keras.layers.Conv2D(filters=64, activation=tf.nn.relu,kernel_size=5),
    tf.keras.layers.MaxPool2D(pool_size=2),
    tf.keras.layers.UpSampling2D(size=2),
    tf.keras.layers.Conv2DTranspose(filters=64, activation=tf.nn.relu,kernel_size=5),
    tf.keras.layers.Conv2DTranspose(filters=64, activation=tf.nn.relu,kernel_size=5),
    tf.keras.layers.UpSampling2D(size=3),
    tf.keras.layers.Conv2DTranspose(filters=32, activation=tf.nn.relu,kernel_size=7),
    tf.keras.layers.Dense(units = 3, activation=tf.nn.relu)
    ])

    for file in os.listdir(train_dir) :
            image = Image.open(train_dir  +r"\\" + file).convert("RGB")
            image = image.resize((120, 120), Image.ANTIALIAS)
            x = np.array(image)
            x = x/255
            X_train.append(x)

    X_train = np.array(X_train)
    autoencoder.compile(optimizer='adam', loss='MSE',metrics=['accuracy'])
    logdir = os.path.join("logs3")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    autoencoder.summary()
    autoencoder.fit(X_train, X_train, epochs=20, callbacks=[tensorboard_callback])
    autoencoder.save(filepath)
    
    #instance = X_train[16]
    #q_values = autoencoder.predict(instance[None])
    #recons = np.array(q_values)*255
    #print(recons.shape)
    #img = Image.fromarray(recons[0], 'RGB')
    #img.show()
