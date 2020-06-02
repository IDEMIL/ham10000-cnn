import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input

train_dir = 'data/train'
valid_dir = 'data/valid'

#Default model is MobileNet
model_path = 'models\mobilenet0.25_no_top.h5' 

# Load the training and validation sets into Dataset objects
imageDataGenerator = ImageDataGenerator(preprocessing_function=preprocess_input)
train_data = imageDataGenerator.flow_from_directory(train_dir, target_size=(224, 224), classes=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'], batch_size=10)
valid_data = imageDataGenerator.flow_from_directory(valid_dir, target_size=(224, 224), classes=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'], batch_size=10)

# Load model - Initial Download
# model = MobileNet(alpha=0.25, include_top=False)

# Load model - Local Directory
model = load_model(model_path)

model.summary()

print(len(model.layers))

#model.save(model_path)