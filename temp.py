import tensorflow as tf

MODEL=tf.keras.models.load_model("my_model.keras")

MODEL.summary()