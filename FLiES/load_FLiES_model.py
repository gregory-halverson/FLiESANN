from os.path import join, abspath, dirname
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow as tf

    # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # tf.disable_v2_behavior()
    # tf.logging.set_verbosity(tf.logging.ERROR)
    # from keras.engine.saving import load_model
    from keras.models import load_model
    from keras.saving import register_keras_serializable

DEFAULT_MODEL_FILENAME = join(abspath(dirname(__file__)), "FLiESANN.h5")

@register_keras_serializable()
def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def load_FLiES_model(model_filename: str = DEFAULT_MODEL_FILENAME):
    return load_model(model_filename, custom_objects={'mae': mae})
