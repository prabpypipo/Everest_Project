from deepface.detectors import OpenCvWrapper, SsdWrapper, DlibWrapper, MtcnnWrapper, RetinaFaceWrapper
from PIL import Image
import math
import numpy as np
import cv2
import os
import pandas as pd

def MultiDetect_face(detector, img, align = True):
    detected_face = None
    img_region = [0, 0, img.shape[0], img.shape[1]]

    faces = []
    try:
    	faces = detector["face_detector"].detectMultiScale(img, 1.1, 10)
    except:
    	pass

    face_set = []
    region_set = []

    for face in faces:
        x,y,w,h = face
        detected_face = img[int(y):int(y +h), int(x):int(x + w)]
        if align:
            detected_face = OpenCvWrapper.align_face(detector["eye_detector"], detected_face)
        img_region = [x, y, w, h]

        face_set.append(detected_face)
        region_set.append(img_region)

    return face_set , region_set


def detect_face(face_detector, detector_backend, img, align = True):

    backends = {
        'opencv': MultiDetect_face,
        'ssd': SsdWrapper.detect_face,
        'dlib': DlibWrapper.detect_face,
        'mtcnn': MtcnnWrapper.detect_face,
        'retinaface': RetinaFaceWrapper.detect_face
    }

    detect_face = backends.get(detector_backend)

    if detect_face:
        face_set, region_set = detect_face(face_detector, img, align)
    else:
        raise ValueError("invalid detector_backend passed - " + detector_backend)

    return face_set, region_set