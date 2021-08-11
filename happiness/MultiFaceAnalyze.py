import pandas as pd
from pathlib import Path
import numpy as np

from deepface.commons import functions
from tqdm import tqdm
from deepface.DeepFace import build_model
import cv2

from oracle.udf.happiness  import preprocess_face

def conver_to_char(i):
    name = ""
    while i != 0 :
        k = i % 10
        i //= 10
        k = chr(k + 48)
        name = k + name
    return name

def analyze(img_path, actions = ['emotion'] , models = {}, enforce_detection = True, detector_backend = 'opencv', prog_bar = True):


    img_paths, bulkProcess = functions.initialize_input(img_path)

    #---------------------------------

    built_models = list(models.keys())

    #---------------------------------

    #pre-trained models passed but it doesn't exist in actions
    if len(built_models) > 0:
        if 'emotion' in built_models and 'emotion' not in actions:
            actions.append('emotion')

        if 'age' in built_models and 'age' not in actions:
            actions.append('age')

        if 'gender' in built_models and 'gender' not in actions:
            actions.append('gender')

        if 'race' in built_models and 'race' not in actions:
            actions.append('race')

    #---------------------------------

    if 'emotion' in actions and 'emotion' not in built_models:
        models['emotion'] = build_model('Emotion')

    if 'age' in actions and 'age' not in built_models:
        models['age'] = build_model('Age')

    if 'gender' in actions and 'gender' not in built_models:
        models['gender'] = build_model('Gender')

    if 'race' in actions and 'race' not in built_models:
        models['race'] = build_model('Race')

    #---------------------------------

    resp_objects = []

    disable_option = (False if len(img_paths) > 1 else True) or not prog_bar

    global_pbar = tqdm(range(0,len(img_paths)), desc='Analyzing', disable = disable_option)

    for j in global_pbar:
        img_path = img_paths[j]

        img_set, region_set = preprocess_face.preprocess_face(img = img_path, target_size = (48, 48), grayscale = True, enforce_detection = enforce_detection, detector_backend = detector_backend, return_region = True)

        resp_obj = {}

        disable_option = (False if len(actions) > 1 else True) or not prog_bar

        pbar = tqdm(range(0, len(img_set)), desc='Finding actions', disable = disable_option)

        img_224 = None # Set to prevent re-detection

        region = [] # x, y, w, h of the detected face region
        region_labels = ['x', 'y', 'w', 'h']

        is_region_set = False

        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        #facial attribute analysis
        for index in pbar:
            img = img_set[index]
            region = region_set[index]

            pbar.set_description("Face: %s" % (index))
                
            emotion_predictions = models['emotion'].predict(img)[0,:]

            sum_of_predictions = emotion_predictions.sum()

            name = 'Face ' + conver_to_char(index + 1)
            resp_obj[name] = {}

            for i in range(0, len(emotion_labels)):
                emotion_label = emotion_labels[i]
                emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
                resp_obj[name][emotion_label] = emotion_prediction

            resp_obj[name]["dominant_emotion"] = emotion_labels[np.argmax(emotion_predictions)]

        #-----------------------------
            for i, parameter in enumerate(region_labels):
                resp_obj[name][parameter] = int(region[i]) #int cast is for the exception - object of type 'float32' is not JSON serializable

        #---------------------------------

        if bulkProcess == True:
            resp_objects.append(resp_obj)
        else:
            return resp_obj

    if bulkProcess == True:

        resp_obj = {}

        for i in range(0, len(resp_objects)):
            resp_item = resp_objects[i]
            resp_obj["instance_%d" % (i+1)] = resp_item

        return resp_obj
