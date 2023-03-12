
import tensorflow as tf
import random



def random_mix(postensor, negtensor, data, label):
    #if less than zero, go for neg; greater then zero, go for pos
    for i in range(len(postensor)+len(negtensor)):
        rn = random.random()
        if len(postensor) == 0:
            length = len(negtensor)
            label += [1 for i in range(length)]
            data += negtensor
            break;
        if len(negtensor) == 0:
            length = len(postensor)
            label += [-1 for i in range(length)]
            data += postensor
            break;
        if rn > 0.5:
            data.append(postensor[-1])
            label.append(1)
            postensor.pop()
        else:
            data.append(negtensor[-1])
            label.append(-1)
            negtensor.pop()

negtensor = list(tf.io.parse_tensor(tf.io.read_file("negtensor.txt"),out_type=tf.float32))
postensor = list(tf.io.parse_tensor(tf.io.read_file("postensor.txt"),out_type=tf.float32))


data = []
label = []

random_mix(postensor,negtensor,data,label)
print(len(data))
print(len(data[0]))
print(type(data[0]))
print(len(label))

data = tf.convert_to_tensor(data, dtype=tf.float32)
label = tf.convert_to_tensor(label,dtype=tf.float32)
tf.io.write_file("mixedinput.txt", tf.io.serialize_tensor(data))
tf.io.write_file("inputlabel.txt", tf.io.serialize_tensor(label))
