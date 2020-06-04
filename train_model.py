import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input

# This is the directory where the trained model will be saved
trained_model_dir = 'data/trained'

train_dir = 'data/train'
valid_dir = 'data/valid'

#Default model is MobileNet
model_path = 'models\mobilenet0.25_no_top.h5' 

batch_size_train = 8
batch_size_valid = 8

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
x = Dropout(0.25)(x)
predictions = Dense(7, activation='softmax')(x)
model = Model(inputs=mobileNet.input, outputs=predictions)

# Set all the layers except the last as trainable: This can be changed in the future
for layer in model.layers[:-6]:
	layer.trainable = False
	
model.summary()

# Compile the model
model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

train_log = model.fit(train_data, steps_per_epoch=np.ceil(6000/8), validation_data=valid_data, validation_steps=np.ceil(2000/8), epochs=25, verbose=1)

model.save(trained_model_dir + '/mobileNet_0.25_finalLayerTrained.h5')
