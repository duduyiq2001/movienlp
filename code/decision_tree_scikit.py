import tensorflow as tf

import numpy as np
from sklearn.metrics import zero_one_loss as J01

import sklearn.tree as tree

from joblib import dump, load




Xtr= tf.io.parse_tensor(tf.io.read_file("mixedinputbow.txt"),out_type=tf.int32)
Ytr = tf.io.parse_tensor(tf.io.read_file("inputlabelbow.txt"),out_type=tf.int32)
Xte = tf.io.parse_tensor(tf.io.read_file("mixedinputbowtest.txt"),out_type=tf.int32)
Yte = tf.io.parse_tensor(tf.io.read_file("inputlabelbowtest.txt"),out_type=tf.int32)

learner = tree.DecisionTreeClassifier(max_depth = 30,criterion='entropy', min_samples_leaf= 30)
learner.fit(Xtr, Ytr)
dump(learner, "decisionTreeScikit.joblib")
print(f'Training error rate: {J01(learner.predict(Xtr),Ytr)}')
print(f'Validation error rate: {J01(learner.predict(Xte),Yte)}')
