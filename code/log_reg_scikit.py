from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import zero_one_loss as J01
from joblib import dump, load
import tensorflow as tf
inputdata = tf.io.parse_tensor(tf.io.read_file("mixedinput1.txt"),out_type=tf.float32)
inputlabel = tf.io.parse_tensor(tf.io.read_file("inputlabel1.txt"),out_type=tf.float32)
Xte = tf.io.parse_tensor(tf.io.read_file("mixedinputtest.txt"),out_type=tf.float32)
Yte = tf.io.parse_tensor(tf.io.read_file("inputlabeltest.txt"),out_type=tf.float32)
inputlabel1 = [1 if a == 1 else 0 for a in inputlabel]
inputlabel1 = tf.convert_to_tensor(inputlabel1, dtype=tf.int32)
Yte= [1 if a == 1 else 0 for a in inputlabel]
Yte = tf.convert_to_tensor(Yte, dtype=tf.int32)

Xtr = inputdata
Ytr = inputlabel1

clf = LogisticRegression(max_iter = 60)
clf.fit(Xtr,Ytr)
dump(clf, "LoregScikitbow.joblib")
print(clf.predict(Xtr)[0:20])
print(f'Training error rate: {J01(clf.predict(Xtr),Ytr)}')
print(f'Validation error rate: {J01(clf.predict(Xte),Yte)}')

