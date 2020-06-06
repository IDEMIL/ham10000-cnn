import tensorflow as tf
import numpy as np
import pickle

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input

import matplotlib.pyplot as plt

# This is the directory where the trained model will be saved
trained_model_dir = 'models/trained/mobilenet_retrainedpt2.h5'

train_dir = 'data/train'
valid_dir = 'data/valid'

#Default model is MobileNet
model_path = 'models\mobilenet0.25_no_top.h5' 

batch_size_train = 32
batch_size_valid = 32

# Load the training and validation sets into Dataset objects
imageDataGenerator = ImageDataGenerator(preprocessing_function=preprocess_input)
train_data = imageDataGenerator.flow_from_directory(train_dir, target_size=(224, 224), classes=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'], batch_size=batch_size_train)
valid_data = imageDataGenerator.flow_from_directory(valid_dir, target_size=(224, 224), classes=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'], batch_size=batch_size_valid)

# Load model - Initial Download
mobileNet = MobileNet()

# Load model - Local Directory
#mobileNet = load_model(model_path)

mobileNet.summary()

print(len(mobileNet.layers))

#mobileNet.save(model_path)

# Here we configure and create a new model from the existing one
x = mobileNet.layers[-6].output
x = Dropout(0.20)(x)
predictions = Dense(7, activation='softmax')(x)
model = Model(inputs=mobileNet.input, outputs=predictions)


# Set all the layers except the last 6 as trainable: This can be changed in the future
for layer in model.layers[:-6]:
	layer.trainable = False
	
model.summary()

# Compile the model
model.compile(Adam(lr=0.003), loss='categorical_crossentropy', metrics=['accuracy'])
epoch_steps = np.floor(6000 / batch_size_train)
valid_steps = np.floor(2000/ batch_size_valid)

hist = model.fit(train_data, steps_per_epoch=epoch_steps, validation_data=valid_data, validation_steps=valid_steps, callbacks=[ReduceLROnPlateau()], epochs=50, verbose=1)

# Save trained model
model.save(trained_model_dir)

# Save history for later use
with open('/trainHistoryDictMobileNetpt1', 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)

train_loss = hist.history['loss']
val_loss = hist.history['val_loss']
train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
epochs = range(1, len(train_acc) + 1)

plt.figure(1, figsize=(7,5))
plt.plot(epochs, train_loss, 'go')
plt.plot(epochs, val_loss)
plt.xlabel('No. of epochs')
plt.ylabel('Loss')
plt.title('validation and training loss')
plt.grid(True)
plt.legend(['Training loss', 'Validation loss'])
plt.style.use(['classic'])

plt.figure(2, figsize=(7,5))
plt.plot(epochs, train_acc, 'go')
plt.plot(epochs, val_acc)
plt.xlabel('No. of epochs')
plt.ylabel('Accuracy')
plt.title('validation and training accuracy')
plt.grid(True)
plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
plt.style.use(['classic'])

plt.show()
