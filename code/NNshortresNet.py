import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add

#contruct model
featureinput = Input(shape = (384,))
input2 = Dense(500,activation = 'relu')(featureinput)
input25 = Dropout(rate = 0.2)(input2)
input3 = Dense(400,activation = 'relu')(input25)
input4 = Dense(384,activation = 'relu')(input3)
input5 = Add()([featureinput,input4])
output = Dense(1, activation = 'sigmoid')(input5)
model = tf.keras.models.Model(inputs=featureinput, outputs=output)
opti = Adam(learning_rate = 0.00001)
model.compile(loss = 'binary_crossentropy',optimizer = opti,metrics = ['accuracy'])



inputdata = tf.io.parse_tensor(tf.io.read_file("mixedinput.txt"),out_type=tf.float32)
inputlabel = tf.io.parse_tensor(tf.io.read_file("inputlabel.txt"),out_type=tf.float32)
inputlabel1 = [1 if a == 1 else 0 for a in inputlabel]
inputlabel1 = tf.convert_to_tensor(inputlabel1, dtype=tf.int32)
train_size = int(0.8*len(inputdata))


Xval = inputdata[train_size:]
Yval = inputlabel1[train_size:]

Xtr = inputdata[0:train_size]
Ytr = inputlabel1[0:train_size]


early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model.fit(Xtr, Ytr, validation_data=(Xval, Yval),callbacks = [early_stop_callback], epochs = 30,batch_size = 32)
model.summary()
for i, loss in enumerate(model.history.history['loss']):
    print(f"Epoch {i}: Training loss = {loss}")

model.save("NN_resnetwithdropout")
