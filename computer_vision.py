import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# Load the fashion mnst dataset
fmnist = tf.keras.datasets.fashion_mnist

#Load the trining and test split of the fashion MNIST dataset
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

# Lets print the training model
# you can put between 0 to 599999 here
index = 0

# Set number of characters  per row when printing
np.set_printoptions(linewidth=320)

# Print the label and image 
print(f'LABEL: {training_labels[index]}')
print(f'\nIMAGE PIXEL ARRAY:\n {training_images[index]}')

# Visualise the image
plt.imshow(training_images[index])

# Normalize the pixel values of the train and test images 
training_images = training_images / 255.0
test_images = test_images / 255.0

# Now lets design the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Build the model

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])