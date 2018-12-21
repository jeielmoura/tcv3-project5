import numpy as np
import cv2
import os

from constants import *

# ---------------------------------------------------------------------------------------------------------- #
#	Description:                                                                                             #
#		Load all test images from a dataset. Train and test folders must have the same directory structure,  #
#		otherwise labels and their respective indexes will be misaligned.									 #
#		All images must have the same size, the same number of channels and 8 bits per channel.              #
#	Parameters:                                                                                              #
#         path - path to the main folder                                                                     #
#         height - number of image rows                                                                      #
#         width - number of image columns                                                                    #
#         num_channels - number of image channels                                                            #
#	Return values:                                                                                           #
#         X - ndarray with all images                                                                        #
# ---------------------------------------------------------------------------------------------------------- #
def load_test_dataset(path=TEST_FOLDER, height=IMAGE_HEIGHT, width=IMAGE_WIDTH, num_channels=NUM_CHANNELS):
	images = sorted(os.listdir(path))

	num_images = len(images)
	X = np.empty([num_images, height, width, num_channels], dtype=np.uint8)

	for i in range(num_images):
		img = cv2.imread(path + '/' +images[i], cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)
		img = img.reshape(height, width, num_channels)
		assert img.shape == (height, width, num_channels), "%r has an invalid image size!" % images[i]
		assert img.dtype == np.uint8, "%r has an invalid pixel format!" % images[i]
		X[i] = img

	return X

# ---------------------------------------------------------------------------------------------------------- #
# Description:                                                                                               #
#         Load all images from a multiclass dataset (folder of folders). Each folder inside the main folder  #
#         represents a different class and its name is used as class label. Train and test folders must have #
#         the same directory structure, otherwise labels and their respective indexes will be misaligned.    #
#         All images must have the same size, the same number of channels and 8 bits per channel.            #
# Parameters:                                                                                                #
#         path - path to the main folder                                                                     #
#         height - number of image rows                                                                      #
#         width - number of image columns                                                                    #
#         num_channels - number of image channels                                                            #
# Return values:                                                                                             #
#         X - ndarray with all images                                                                        #
#         y - ndarray with indexes of labels (y[i] is the label for X[i])                                    #
#         l - list of existing labels (1st label in the list has index 0, 1nd has index 1, and so on)        #
# ---------------------------------------------------------------------------------------------------------- #
def load_train_dataset(path=TRAIN_FOLDER, height=IMAGE_HEIGHT, width=IMAGE_WIDTH, num_channels=NUM_CHANNELS):
	classes = sorted(os.listdir(path))
	images = [sorted(os.listdir(path+'/'+id)) for id in classes]

	num_images = np.sum([len(l) for l in images])
	X = np.empty([num_images, height, width, num_channels], dtype=np.uint8)

	k = 0
	for i in range(len(classes)):
		for j in range(len(images[i])):
			img = cv2.imread(path+'/'+classes[i]+'/'+images[i][j], cv2.IMREAD_GRAYSCALE)
			img = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)
			img = img.reshape(height, width, num_channels)
			assert img.shape == (height, width, num_channels), "%r has an invalid image size!" % images[i][j]
			assert img.dtype == np.uint8, "%r has an invalid pixel format!" % images[i][j]
			X[k] = img
			k += 1

	return X

X_semi = load_test_dataset()/255.
X_data = load_train_dataset()/255.

X_data = np.concatenate( (X_data, X_semi), axis=0)