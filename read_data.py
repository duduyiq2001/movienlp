import tensorflow as tf

negtensor = tf.io.parse_tensor(tf.io.read_file("negtensor.txt"),out_type=tf.float32)
postensor = tf.io.parse_tensor(tf.io.read_file("postensor.txt"),out_type=tf.float32)
print("negtensor:", negtensor)
print("postensor:", postensor)