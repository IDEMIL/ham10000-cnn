import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input

# This is the path from where the trained model will be loaded from
trained_model_dir = 'models/trained/mobilenet_retrainedpt2.h5'

# This is the directory containing the test data
test_dir = 'data/test'

# Load model
model = load_model(trained_model_dir)

# Load test data
imageDataGenerator = ImageDataGenerator(preprocessing_function=preprocess_input)
test_data = imageDataGenerator.flow_from_directory(test_dir, target_size=(224, 224), classes=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'], batch_size=10, shuffle=False)

results = model.evaluate(test_data, batch_size=10)
print('test loss, test acc:', results)

