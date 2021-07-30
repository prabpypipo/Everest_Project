import os
import numpy as np
import pandas as pd
from pathlib import Path
import zipfile
import cv2
import tqdm
import matplotlib.pyplot as plt
from deepface import DeepFace
from happiness import MultiFaceAnalyze

obj_names = ['person']

class Happiness():
    def __init__(self):
        super().__init__()
        self.arg_parser.add_argument("--class_thres", type=float, default=0.5)
        self.arg_parser.add_argument("--obj_thres", type=float, default=0)
        self.arg_parser.add_argument("--obj", type=str, choices=obj_names, default="person")
    
    def get_img_size(self):
        return (224, 224)
    
    def get_scores(self, imgs, visualize=False):
        obj=MultiFaceAnalyze.analyze(imgs)
        #print(obj["age"]," years old ",obj["dominant_race"]," ",obj["dominant_emotion"]," ", obj["gender"])
        #new = pd.DataFrame.from_dict(obj) 
        #print(type(new))
        emotions = ["angry" , "disgust" , "fear" , "happy" , "sad" , "surprise" , "neutral"]
        weights = [-1, -1, -1, 1, -1, 1, 0]
        sum_happiness = 0
        #print(obj)
        for face in obj.keys():
            dict_emotions = obj[face]
            for i in range(7):
                emotion = emotions[i]
                weight = weights[i]
                sum_happiness += dict_emotions[emotion] * weight

        #print("sum = ",sum_happiness)
        #print('average = ', sum_happiness / len(obj.keys()))

        return sum_happiness / len(obj.keys())