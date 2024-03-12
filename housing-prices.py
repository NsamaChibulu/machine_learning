import tensorflow as tf 
import numpy as np
from tensorflow import keras 

def house_model():
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=int)
    ys = np.array([100000.0, 150000.0, 200000.0, 250000.0, 300000.0, 350000.0], dtype=int)
    

    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer="sgd", loss="mean_squared_error")

    model.fit(xs, ys, epochs=500)
    model.predict([10.0])

    return model
model = house_model()

new_x = 7.0
prediction = model.predict([new_x])[0]
print(prediction)

