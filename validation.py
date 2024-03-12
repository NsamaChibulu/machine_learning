import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# Load the datasets from the interned.
# train_df = contains the training set
# test_df = contains the test set

train_df = pd.read_csv("california_housing_train.csv")
test_df = pd.read_csv("california_housing_test.csv")

# The following code scales the median_house_value.
scale_factor = 1000.0

# Scale the training set labels.
train_df["median_house_value"] /= scale_factor

# Scale the test set's label
test_df["median_house_value"] /= scale_factor

# Load the funcitons that build and train the model.

def build_model(my_learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))
    model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    
    return model

def train_model(model, df, feature, label, my_epochs,
                my_batch_size=None, my_validation_split=0.1):
    """Feeed a dataset into the model in order to train it"""
    history = model.fit(x=df[feature],
                        y=df[label],
                        batch_size=my_batch_size,
                        epochs=my_epochs,
                        validation_split=my_validation_split)
    # Gather the models trained weight and bias
    trained_weight = model.get_weights()[0][0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored seperately from the rest of history
    epochs = history.epoch

    #Isolate the root mean squared error for each epoch
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    return epochs, rmse, history.history

print('Defined the build and train model')
