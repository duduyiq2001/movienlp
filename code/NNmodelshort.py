import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import pandas as pd


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

NNmodel = Sequential()
NNmodel.add(Dense(500, activation = 'relu'))
NNmodel.add(Dropout(rate = 0.2))
NNmodel.add(Dense(200, activation = 'relu'))
NNmodel.add(Dense(1, activation = 'sigmoid'))
opti = Adam(learning_rate = 0.00001)
NNmodel.compile(loss = 'binary_crossentropy',optimizer = opti,metrics = ['accuracy'])

inputdata = tf.io.parse_tensor(tf.io.read_file("mixedinput1.txt"),out_type=tf.float32)
inputlabel = tf.io.parse_tensor(tf.io.read_file("inputlabel1.txt"),out_type=tf.float32)
inputlabel1 = [1 if a == 1 else 0 for a in inputlabel]
inputlabel1 = tf.convert_to_tensor(inputlabel1, dtype=tf.int32)
train_size = int(0.8*len(inputdata))


Xval = inputdata[train_size:]
Yval = inputlabel1[train_size:]

Xtr = inputdata[0:train_size]
Ytr = inputlabel1[0:train_size]

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)


NNmodel.fit(Xtr, Ytr, validation_data=(Xval, Yval),callbacks = [early_stop_callback], epochs = 30,batch_size = 32)
NNmodel.summary()
for i, loss in enumerate(NNmodel.history.history['loss']):
    print(f"Epoch {i}: Training loss = {loss}")




# Export the model to a SavedModel.
NNmodel.save("NN_reducedlayers")

print(NNmodel.predict(inputdata[0:20]))
