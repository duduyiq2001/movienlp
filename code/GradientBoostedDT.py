import tensorflow as tf

print(tf.__version__)
import tensorflow_decision_forests as tfdf
from tensorflow.keras.callbacks import EarlyStopping
print(tf.__version__)
num_trees = 1000
max_depth = 8
learning_rate = 0.01
subsample_rate = 0.7
subsample_features = 0.6
l1_reg = 0.01
l2_reg = 0.01

gtbt_model = tfdf.keras.GradientBoostedTreesModel(num_trees = num_trees, max_depth = max_depth, shrinkage = learning_rate,
                                                 subsample = subsample_rate, num_candidate_attributes_ratio = subsample_features,check_dataset = False,loss = "BINOMIAL_LOG_LIKELIHOOD",sampling_method = "RANDOM",early_stopping= 'LOSS_INCREASE')

#load data
inputdata = tf.io.parse_tensor(tf.io.read_file("/content/movienlp/mixedinput1.txt"),out_type=tf.float32)
inputlabel = tf.io.parse_tensor(tf.io.read_file("/content/movienlp/inputlabel1.txt"),out_type=tf.float32)
inputlabel1 = [1 if a == 1 else 0 for a in inputlabel]
inputlabel1 = tf.convert_to_tensor(inputlabel1, dtype=tf.int32)
val_size = int(0.2*len(inputdata))

'''
dataset = tf.data.Dataset.from_tensor_slices((inputdata,inputlabel))
dataset = dataset.shuffle(len(dataset))
val_size = int(0.2*len(dataset))
trainset = dataset.skip(val_size)
valset = dataset.take(val_size)
print(trainset.element_spec)
print(valset.element_spec)
'''
early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)


gtbt_model.fit(inputdata[0:val_size], inputlabel1[0:val_size], validation_data=(inputdata[val_size:], inputlabel1[val_size:]) )
print("finished training!")
gtbt_model.summary()

gtbt_model.save("/content/movienlp/GBDT")


