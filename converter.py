import tensorflow as tf

saved_model_dir = './saved_model/model'

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=saved_model_dir)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

