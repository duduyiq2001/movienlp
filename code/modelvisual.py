import tensorflow as tf

new_model = tf.keras.models.load_model('../model/NN_longbow')
tf.keras.utils.plot_model(
    model = new_model,
    to_file='model.png',
    show_shapes=False,
    show_dtype=False,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=False,
    dpi=96,
    layer_range=None,
    show_layer_activations=True
)