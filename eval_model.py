import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from efficientnet.tfkeras import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import itertools
import numpy as np

# This is the path from where the trained model will be loaded from
trained_model_dir = 'models/trained/effnetb4_retrainedpt9.h5'

# This is the directory containing the test data
test_dir = 'data/test'

# Load model
model = load_model(trained_model_dir)

# Load test data
imageDataGenerator = ImageDataGenerator(preprocessing_function=preprocess_input)
test_data = imageDataGenerator.flow_from_directory(test_dir, target_size=(380, 380), classes=['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'], batch_size=1, shuffle=False)

results = model.evaluate(test_data, batch_size=10)
print('test loss, test acc:', results)

test_labels = test_data.labels

print(test_labels)

predictions = model.predict(test_data, steps=2003, verbose=1)
rounded_preds = np.argmax(predictions, axis=-1)

print(rounded_preds)

matrix = confusion_matrix(test_labels, rounded_preds)

matrix_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(matrix, matrix_labels)
plt.show()

evaluation = classification_report(rounded_preds, test_data.classes, target_names=matrix_labels)

print(evaluation)
