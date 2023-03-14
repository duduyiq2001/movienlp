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
input3 = Dense(400,activation = 'relu')(input2)
input4 = Dense(384,activation = 'relu')(input3)
input5 = Add()([featureinput,input4])
output = Dense(1, activation = 'sigmoid')(input5)
model = tf.keras.models.Model(inputs=featureinput, outputs=output)
opti = Adam(learning_rate = 0.00001)
model.compile(loss = 'binary_crossentropy',optimizer = opti,metrics = ['accuracy'])

'''
x1 = tf.keras.layers.Dense(8, activation='relu')(input1)
input2 = tf.keras.layers.Input(shape=(32,))
x2 = tf.keras.layers.Dense(8, activation='relu')(input2)
# equivalent to `added = tf.keras.layers.add([x1, x2])`
added = tf.keras.layers.Add()([x1, x2])
out = tf.keras.layers.Dense(4)(added)
model = tf.keras.models.Model(inputs=[input1, input2], outputs=out) in the second last line, is it passing the output of the add as the input of the dense layer?
'''

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
    patience=5,
    restore_best_weights=True
)

model.fit(inputdata[0:val_size], inputlabel1[0:val_size], validation_data=(inputdata[val_size:], inputlabel1[val_size:]),callbacks = [early_stop_callback], epochs = 30,batch_size = 32)
model.summary()
for i, loss in enumerate(model.history.history['loss']):
    print(f"Epoch {i}: Training loss = {loss}")

model.save("NN_resnet")