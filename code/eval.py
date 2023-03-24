#for evaluating different models
import tensorflow as tf
from sklearn.metrics import zero_one_loss
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
new_model = tf.keras.models.load_model('NN_longresnetwwithtworesblocksBOW')

testdata = tf.io.parse_tensor(tf.io.read_file("mixedinputbowtest.txt"),out_type=tf.int32)
testlabel = tf.io.parse_tensor(tf.io.read_file("inputlabelbowtest.txt"),out_type=tf.int32)
testlabel1 = [1 if a == 1 else 0 for a in testlabel]
print(len(testlabel1))
testlabel1 = tf.convert_to_tensor(testlabel1, dtype=tf.int32)
predict1 = new_model.predict(testdata)
predict1 = [1 if a >=0.5 else 0 for a in predict1]
print(zero_one_loss(testlabel1,predict1))
