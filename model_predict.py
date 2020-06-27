import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input
from keras.preprocessing import image
import numpy as np

# This is the path from where the trained model will be loaded from
trained_model_dir = 'models/trained/mobilenet_retrainedpt2.h5'

# This is the directory containing the test data
test_dir = 'data/test'

# Load model
model = load_model(trained_model_dir)

img = image.load_img('data/test/nv/ISIC_0024335.jpg', target_size = (224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)

classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

resultset = model.predict(img)

prediction_string = ''
index = 0
for result in resultset[0]:
	prediction_string += '{} {}\n'.format(classes[index], result)
	index += 1



print(prediction_string)