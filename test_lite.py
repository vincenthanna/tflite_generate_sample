import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

print("x_test.shape =", x_test.shape)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
print("input_shape =", input_shape)

acc_cnt = 0
for i in range(x_test.shape[0]):
    x = x_test[i]
    y = y_test[i]
    x = x[None, ...] # (1, 28, 28, 1) 로 변환
    x = x.astype(np.float32) # type을 맞춰줘야 한다.

    # set data to interpreter
    interpreter.set_tensor(input_details[0]['index'], x)

    # run inference
    interpreter.invoke()

    # get result from output_details
    output_data = interpreter.get_tensor(output_details[0]['index'])    
    output_data = tf.argmax(output_data, axis=1) # convert to index(number)

    if output_data == y:
        acc_cnt = acc_cnt + 1
    
print(f"Test Data Accuracy from TFLite Model : {acc_cnt / x_test.shape[0]:.4f}")


# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data)
