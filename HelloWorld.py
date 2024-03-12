## Neural Network
 
import tensorflow as tf
import numpy as np
from tensorflow import keras

## Define and Compile the neural network
# Build a simple Sequential model
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Now compile the neural network model
model.compile(optimizer="sgd",
              loss="mean_squared_error")

# Declare model inputs and outputs for training
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train the model
model.fit(xs, ys, epochs=500)

# Make a prediction 
print(model.predict([10.0]) )
