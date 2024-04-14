import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow_decision_forests as tfdf
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras
import math

archive_path = tf.keras.utils.get_file("letor.zip",
  "https://download.microsoft.com/download/E/7/E/E7EABEF1-4C7B-4E31-ACE5-73927950ED5E/Letor.zip",
  extract=True)

# Path to a ranking dataset using libsvm format

raw_dataset_path = os.path.join(os.path.dirname(archive_path),"OHSUMED/Data/Fold1/trainingset.txt")

# Next step is to convert the dataset into a flt CSV
def convert_libsvm_to_csv(src_path, dst_path):
    ''' this code is specific to the set'''
    dst_handle = open(dst_path, "w")
    first_line = True
    for src_line in open(src_path, "r"):
        # Note: the last 3 items are comments
        items = src_line.split(" ")[:-3]
        relevance = items[0]
        group = items[1].split(":")[1]
        features = [ items[1].split(":") for item in items [2:]]

        if first_line:
            # CSV header 
            dst_handle.write("relevance,group," + ",".join(["f_" + feature[0] for feature in features]) + "\n")
            first_line = False
        dst_handle.write(relevance + ",g_" + group + "," + (",".join([feature[1] for feature in features])) + "\n")
    dst_handle.close()


# Convert the dataset
csv_dataset_path = "ohsumed.csv"
convert_libsvm_to_csv(raw_dataset_path, csv_dataset_path)

# Load a dataset into a pandas dataframe
dataset_df = pd.read_csv(csv_dataset_path)

# Display the first 3 examples
print(dataset_df.head(3))


# Convert panadas dataframe into a tensorflow dataset
dataset_ds = tfdf.keras.pd_dataframe_to_tf_dataset(dataset_df, label="relevance", task=tfdf.keras.task.RANKING)

# Now lets configure and train our ranking model
model = tfdf.keras.GradientBoostedTreesModel(
    task=tfdf.keras.Task.RANKING,
    ranking_group="group",
    num_tress=50
)

model.fit(dataset_ds)
