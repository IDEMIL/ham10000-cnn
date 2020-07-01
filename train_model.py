import tensorflow as tf
import numpy as np
import pickle

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from efficientnet.tfkeras import EfficientNetB4
from efficientnet.tfkeras import preprocess_input

import matplotlib.pyplot as plt

import os.path

# This is the directory where the trained model will be saved
trained_model_dir = 'models/trained/effnetb4_retrainedpt12.h5'

if os.path.exists(trained_model_dir):
	print('Error: Please specify a new path for the newly trained model in the source code.')
	exit()

# These directories contain the training and validation data
train_dir = 'data/train'
valid_dir = 'data/valid'

# Batch size was chosen with help from https://stackoverflow.com/questions/49922252/choosing-number-of-steps-per-epoch
batch_size_train = 12
batch_size_valid = 12

# Load the training and validation sets into Dataset objects
train_imageDataGenerator = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
test_imageDataGenerator = ImageDataGenerator(preprocessing_function=preprocess_input)
train_data = train_imageDataGenerator.flow_from_directory(train_dir, target_size=(380, 380), classes=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'], batch_size=batch_size_train)
valid_data = test_imageDataGenerator.flow_from_directory(valid_dir, target_size=(380, 380), classes=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'], batch_size=batch_size_valid)


# Load model - Initial Download
effnet = EfficientNetB4(weights='imagenet')

effnet.summary()

print(len(effnet.layers))

# Here we configure and create a new model from the existing one
x = effnet.layers[-3].output
x = Dropout(0.30)(x)
predictions = Dense(7, activation='softmax')(x)
model = Model(inputs=effnet.input, outputs=predictions)

# Set all the layers except the last 141 as trainable: This can be changed in the future
for layer in model.layers[:-141]:
	layer.trainable = False
	
model.summary()

# Compile the model
model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
epoch_steps = np.floor(6000 / batch_size_train)
valid_steps = np.floor(2000/ batch_size_valid)
# Save the model on the epoch where it produces the minimum loss on the validation set
checkpoint = ModelCheckpoint(trained_model_dir, monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
lrReducer = ReduceLROnPlateau(verbose=1)

hist = model.fit(train_data, steps_per_epoch=epoch_steps, validation_data=valid_data, validation_steps=valid_steps, callbacks=[lrReducer, checkpoint], epochs=40, verbose=1)

# The following code plots validation and loss, code adapted from Anuj Shah at https://www.youtube.com/watch?v=9pDlJ5aAFN4

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
epochs = range(1, len(train_acc) + 1)

plt.figure(1, figsize=(7,5))
plt.plot(epochs, train_loss, color='yellowgreen', linewidth=3)
plt.plot(epochs, val_loss, color='rebeccapurple', linewidth=3)
plt.xlabel('No. of epochs')
plt.ylabel('Loss')
plt.title('validation and training loss')
plt.grid(True)
plt.legend(['Training loss', 'Validation loss'])
plt.style.use(['classic'])

plt.figure(2, figsize=(7,5))
plt.plot(epochs, train_acc, color='yellowgreen', linewidth=3)
plt.plot(epochs, val_acc, color='rebeccapurple', linewidth=3)
plt.xlabel('No. of epochs')
plt.ylabel('Accuracy')
plt.title('validation and training accuracy')
plt.grid(True)
plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
plt.style.use(['classic'])

plt.show()
