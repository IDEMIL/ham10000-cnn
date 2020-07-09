from flask import Flask, jsonify, request
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from efficientnet.tfkeras import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import base64
import io
from PIL import Image


app = Flask(__name__)

# This is the path from where the trained model will be loaded from
trained_model_dir = 'effnetb4_retrainedpt9.h5'
	
def get_model():
	global model
	model = load_model(trained_model_dir)
	print(" * Model loaded.")
	
def preprocess_image(image, target_size):
	img = image.resize(target_size)
	img = img_to_array(img)
	img = preprocess_input(img)
	img = np.expand_dims(img, axis = 0)
	
	return img
	
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
print(" * Loading Keras model...")
get_model()
	

@app.route('/predictions', methods=['POST'])
def predictions():
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	image = Image.open(io.BytesIO(decoded))
	processed_img = preprocess_image(image, target_size=(380, 380))
	
	resultset = model.predict(processed_img)
	
	response = {
		'prediction': {
			classes[0]: str(resultset[0][0]),
			classes[1]: str(resultset[0][1]),
			classes[2]: str(resultset[0][2]),
			classes[3]: str(resultset[0][3]),
			classes[4]: str(resultset[0][4]),
			classes[5]: str(resultset[0][5]),
			classes[6]: str(resultset[0][6])
		}
	}
	return jsonify(response)












