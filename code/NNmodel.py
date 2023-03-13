import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

NNmodel = Sequential()
NNmodel.add(Dense(500, activation = 'relu'))
NNmodel.add(Dropout(rate = 0.2))
NNmodel.add(Dense(400, activation = 'relu'))
NNmodel.add(Dense(300, activation = 'relu'))
NNmodel.add(Dense(200, activation = 'relu'))
NNmodel.add(Dense(100, activation = 'relu'))
NNmodel.add(Dense(1, activation = 'sigmoid'))
NNmodel.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

inputdata = tf.io.parse_tensor(tf.io.read_file("mixedinput1.txt"),out_type=tf.float32)
inputlabel = tf.io.parse_tensor(tf.io.read_file("inputlabel1.txt"),out_type=tf.float32)
inputlabel1 = [1 if a == 1 else 0 for a in inputlabel]
inputlabel1 = tf.convert_to_tensor(inputlabel1, dtype=tf.int32)
val_size = int(0.2*len(inputdata))


Xval = inputdata[0:val_size]
Yval = inputlabel1[0:val_size]

Xtr = inputdata[val_size:]
Ytr = inputlabel1[val_size:]

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)


NNmodel.fit(inputdata[0:val_size], inputlabel1[0:val_size], validation_data=(inputdata[val_size:], inputlabel1[val_size:]),callbacks = [early_stop_callback], epochs = 1,batch_size = 32)
NNmodel.summary()
for i, loss in enumerate(NNmodel.history.history['loss']):
    print(f"Epoch {i}: Training loss = {loss}")



# Evaluate the model.
#model.evaluate(test_ds)

# Export the model to a SavedModel.
NNmodel.save("NN_withdropout")

print(NNmodel.predict(inputdata[0:20]))
