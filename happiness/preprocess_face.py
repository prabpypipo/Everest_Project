import numpy as np
import cv2

from deepface.detectors import FaceDetector
from deepface.commons.functions import load_image

import tensorflow as tf

from tensorflow.keras.preprocessing import image

from oracle.udf.happiness import MultiFaceDetectors


def detect_face(img, detector_backend = 'opencv', grayscale = False, enforce_detection = True, align = True):

	img_region = [0, 0, img.shape[0], img.shape[1]]

	#detector stored in a global variable in FaceDetector object.
	#this call should be completed very fast because it will return found in memory
	#it will not build face detector model in each call (consider for loops)
	face_detector = FaceDetector.build_model(detector_backend)

	face_set, region_set = MultiFaceDetectors.detect_face(face_detector, detector_backend, img, align)

	if len(face_set) == 0 :
		if enforce_detection != True:
 			return face_set, region_set
		else:
			raise ValueError("Face could not be detected. Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")
	return face_set, region_set



def preprocess_face(img, target_size=(224, 224), grayscale = False, enforce_detection = True, detector_backend = 'opencv', return_region = False, align = True):

	#img might be path, base64 or numpy array. Convert it to numpy whatever it is.
	img = load_image(img)
	base_img = img.copy()

	face_set, region_set = detect_face(img = img, detector_backend = detector_backend, grayscale = grayscale, enforce_detection = enforce_detection, align = align)

	#--------------------------

	img_pixels_set = []

	for i in range(len(face_set)):
		img = face_set[i]
		region = region_set[i]

		if img.shape[0] == 0 or img.shape[1] == 0:
			if enforce_detection == True:
				raise ValueError("Detected face shape is ", img.shape,". Consider to set enforce_detection argument to False.")
			else: #restore base image
				img = base_img.copy()

		#--------------------------

		#post-processing
		if grayscale == True:
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
		#---------------------------------------------------
		#resize image to expected shape

		# img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image
	
		# First resize the longer side to the target size
		#factor = target_size[0] / max(img.shape)
	
		factor_0 = target_size[0] / img.shape[0]
		factor_1 = target_size[1] / img.shape[1]
		factor = min(factor_0, factor_1)
	
		dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
		img = cv2.resize(img, dsize)

		# Then pad the other side to the target size by adding black pixels
		diff_0 = target_size[0] - img.shape[0]
		diff_1 = target_size[1] - img.shape[1]
		if grayscale == False:
			# Put the base image in the middle of the padded image
			img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
		else:
			img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')
	
		#double check: if target image is not still the same size with target.
		if img.shape[0:2] != target_size:
			img = cv2.resize(img, target_size)
	
		#---------------------------------------------------
	
		img_pixels = image.img_to_array(img)
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		img_pixels /= 255 #normalize input in [0, 1]

		img_pixels_set.append(img_pixels)

	if return_region == True:
		return img_pixels_set, region_set
	else:
		return img_pixels_set
