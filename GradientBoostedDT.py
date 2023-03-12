import tensorflow as tf
print(tf.__version__)
import tensorflow_decision_forests as tfdf
from tensorflow.keras.callbacks import EarlyStopping
print(tf.__version__)
num_trees = 100
max_depth = 6
learning_rate = 0.1
subsample_rate = 0.7
subsample_features = 0.6
l1_reg = 0.01
l2_reg = 0.01

gtbt_model = tfdf.keras.GradientBoostedTreesModel(num_trees = num_trees, max_depth = max_depth, learning_rate = learning_rate,
                                                 subsample_rate = subsample_rate, subsample_features = subsample_features, l1_regurization=
                                                  l1_reg, l2_regurization = l2_reg, task = "classification",loss = "binary_crossentropy", metrics = "accuracy")

#load data

inputdata = tf.io.parse_tensor(tf.io.read_file("mixedinput1.txt"),out_type=tf.float32)
inputlabel = tf.io.parse_tensor(tf.io.read_file("inputlabel1.txt"),out_type=tf.float32)

dataset = tf.data.Dataset.from_tensor_slices(inputdata,inputlabel)
dataset = dataset.shuffle(len(dataset))
val_size = int(0.2*len(dataset))
trainset = dataset.skip(val_size)
valset = dataset.take(val_size)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

gtbt_model.fit(trainset,validation_data = valset,callbacks = [early_stop_callback] )
print("finished training!")
gtbt_model.summary()

gtbt_model.save("savedmodel/GBDT")


