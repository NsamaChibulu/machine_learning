import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow_decision_forests as tfdf 
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras
import math 


# print("Found TensorFlow Decision Forests v" + tfdf.__version__)

# Here we will train, evaluate, analyse and export a multiclass 
# classification random forest on palmers penguins dataset

# Load a dataset into a Pandas Dataframe.
dataset_df = pd.read_csv("penguins.csv")

# Display the first 3 examples.
print(dataset_df.head(3))

# Keras metrics expect the labels (column headers) to be integrers,
# however they are appearing as strings. So lets convert to integers.

## Encode the the categorical labels as integers
# Name of the label column
label = "species"
classes = dataset_df[label].unique().tolist()
print(f"Label classes: {classes}")

dataset_df[label] = dataset_df[label].map(classes.index)

# Next , split the dataset into a training and a testing dataset

def split_dataset(dataset, test_ratio=0.30):
    """Splits a panda datafram into two"""
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

train_ds_pd, test_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples for testing". format(
    len(train_ds_pd), len(test_ds_pd)))


# Finally convert the pandas dataframe(pd.Dataframe) into tensorfloq
# datasets (tf.data.Dataset)

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds_pd,label=label)

# train the model
# Specify the model
model_1 = tfdf.keras.RandomForestModel(verbose=2)

# Train the model
model_1.fit(train_ds)

#Evaluate the model

model_1.compile(metrics=["accuracy"])
evaluation = model_1.evaluate(test_ds, return_dict=True)
print()

for name, value in evaluation.items():
    print(f"{name}: {value:.4f}")