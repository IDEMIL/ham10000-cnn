import pandas
import numpy as np
import os
import shutil

from efficientnet.tfkeras import preprocess_input

import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

metadata_path = 'data/raw/HAM10000_metadata.csv'

# Read the metadata csv into a pandas dataframe
metadata = pandas.read_csv(metadata_path)

# Display the number of images for each class. Important to show the class imbalance. Adapted from https://pythonspot.com/matplotlib-bar-chart/
print(metadata['dx'].value_counts())

classes = ('nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df')
y_pos = np.arange(len(classes))
performance = metadata['dx'].value_counts()

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, classes)
plt.ylabel('Images')
plt.title('Number of images per class')

# Shuffle the metadata
metadata = metadata.sample(frac=1).reset_index(drop=True)

# Split the data into 60%, 20%, 20% for train, valid, and test respectively
train_metadata, valid_metadata, test_metadata = np.split(metadata, [int(0.6 * len(metadata)), int(0.8 * len(metadata))])

# Open both folders of raw data in a list
images_part1 = os.listdir('data/raw/ham10000_images_part_1')
images_part2 = os.listdir('data/raw/ham10000_images_part_2')

# Convert dataframe to list, to iterate using a for loop
train = list(train_metadata['image_id'])

# iterate through the train dataframe, checking the type of each entry, and transferring it to the appropriate folder

for img in train:
	
	imgFileName = img + '.jpg'
	lesionType = metadata.loc[metadata['image_id'] == img, 'dx'].iloc[0]
	
	if imgFileName in images_part1:
		shutil.copyfile('data/raw/ham10000_images_part_1/' + imgFileName, 'data/train/' + lesionType + '/' + imgFileName)

	if imgFileName in images_part2:
		shutil.copyfile('data/raw/ham10000_images_part_2/' + imgFileName, 'data/train/' + lesionType + '/' + imgFileName)
		
		
valid = list(valid_metadata['image_id'])

# 

for img in valid:
	
	imgFileName = img + '.jpg'
	lesionType = metadata.loc[metadata['image_id'] == img, 'dx'].iloc[0]
	
	if imgFileName in images_part1:
		shutil.copyfile('data/raw/ham10000_images_part_1/' + imgFileName, 'data/valid/' + lesionType + '/' + imgFileName)

	if imgFileName in images_part2:
		shutil.copyfile('data/raw/ham10000_images_part_2/' + imgFileName, 'data/valid/' + lesionType + '/' + imgFileName)
		
		
test = list(test_metadata['image_id'])

# 

for img in test:
	
	imgFileName = img + '.jpg'
	lesionType = metadata.loc[metadata['image_id'] == img, 'dx'].iloc[0]
	
	if imgFileName in images_part1:
		shutil.copyfile('data/raw/ham10000_images_part_1/' + imgFileName, 'data/test/' + lesionType + '/' + imgFileName)

	if imgFileName in images_part2:
		shutil.copyfile('data/raw/ham10000_images_part_2/' + imgFileName, 'data/test/' + lesionType + '/' + imgFileName)
		
		
plt.show()
