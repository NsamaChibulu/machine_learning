import tensorflow as tf
from tensorflow import keras

# Instantiate the dataset API
fmnist = tf.keras.datasets.fashion_mnist

# Load the dataset
(x_train, y_train),(x_test,y_test) = fmnist.load_data()

# normalise the pixels
x_train, x_test = x_train / 255.0, x_test / 255.0


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        """
        halts  the training when the loss falls below 0.4

        Args:
            epoch (integrer) - index of the epoch (required but unused
            in the function definition below)
            logs (dict) - metric results from training epoch
        """
        # Check the loss
        if (logs.get('loss') < 0.4):

            # Stop if the threshold is met
            print("\nLoss is lower than 0.4 so cancelling training")
            self.model.stop_training = True

# Instantiate Class

callbacks = myCallback()

# Define Model

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile model

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])