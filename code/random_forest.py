import tensorflow_decision_forests as tfdf
import tensorflow as tf
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping



inputdata = tf.io.parse_tensor(tf.io.read_file("/content/movienlp/mixedinput1.txt"),out_type=tf.float32)
inputlabel = tf.io.parse_tensor(tf.io.read_file("/content/movienlp/inputlabel1.txt"),out_type=tf.float32)
print(len(inputdata))
print(len(inputlabel))
print(inputlabel[0:30])
inputlabel1 = [1 if a == 1 else 0 for a in inputlabel]
inputlabel1 = tf.convert_to_tensor(inputlabel1, dtype=tf.int32)
print(inputlabel1[0:20])
val_size = int(0.8*len(inputdata))

# Train a Random Forest model.
model = tfdf.keras.RandomForestModel(num_trees=1000, bootstrap_size_ratio = 0.8, max_depth = 8)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Summary of the model structure.
model.fit(inputdata[0:val_size], inputlabel1[0:val_size], validation_data=(inputdata[val_size:], inputlabel1[val_size:]),callbacks = [early_stop_callback])
model.summary()

# Evaluate the model.
#model.evaluate(test_ds)

# Export the model to a SavedModel.
model.save("RANDOM_FOREST")

print(model.predict(inputdata[0:20]))