import tensorflow as tf

import numpy as np
from sklearn.metrics import zero_one_loss as J01

import sklearn.tree as tree

from joblib import dump, load

# Fix the random seed for reproducibility
# !! Important !! : do not change this
seed = 1234
np.random.seed(seed)  

inputdata = tf.io.parse_tensor(tf.io.read_file("mixedinput1.txt"),out_type=tf.float32)
inputlabel = tf.io.parse_tensor(tf.io.read_file("inputlabel1.txt"),out_type=tf.float32)
inputlabel1 = [1 if a == 1 else 0 for a in inputlabel]
inputlabel1 = tf.convert_to_tensor(inputlabel1, dtype=tf.int32)
val_size = int(0.2*len(inputdata))


Xval = inputdata[:val_size, :]
Yval = inputlabel1[:val_size]

Xtr = inputdata[val_size:, :]
Ytr = inputlabel1[val_size:]

learner = tree.DecisionTreeClassifier(max_depth = 7,criterion='entropy', min_samples_leaf= 64, random_state = seed)
learner.fit(Xtr, Ytr)
dump(learner, "decisionTreeScikit.joblib")
print(f'Training error rate: {J01(learner.predict(Xtr),Ytr)}')
print(f'Validation error rate: {J01(learner.predict(Xval),Yval)}')
