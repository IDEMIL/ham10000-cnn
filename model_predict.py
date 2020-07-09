import sys

opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

imgPath = ''

if len(opts) < 1:
	print('No option was specified. Use -h for help.')
	exit()
if opts[0] == '-i':
	args[0] = args[0].replace('\\', '/')
	imgPath = 'data/test/' + args[0]
if opts[0] == '-f':
	args[0] = args[0].replace('\\', '/')
	imgPath = args[0]
if opts[0] == '-h':
	print('-i : Supply a filename and its containining folder for an image contained by the test folder. i.e bcc/ISIC_0024436.jpg')
	print('-f : Supply a full filepath for an image')
	exit()

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from efficientnet.tfkeras import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np



# This is the path from where the trained model will be loaded from
trained_model_dir = 'models/trained/effnetb4_retrainedpt9.h5'

# This is the directory containing the test data
test_dir = 'data/test'

# Load model
model = load_model(trained_model_dir)

img = image.load_img(imgPath, target_size = (380, 380))
img = image.img_to_array(img)
img = preprocess_input(img)
img = np.expand_dims(img, axis = 0)

classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

resultset = model.predict(img)

prediction_string = ''
index = 0
for result in resultset[0]:
	prediction_string += '{} {}\n'.format(classes[index], result)
	index += 1



print(prediction_string)