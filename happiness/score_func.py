import os
import numpy as np
import cv2
from oracle.udf.happiness import MultiFaceAnalyze
from oracle.udf.base import BaseScoringUDF
from PIL import Image

obj_names = ['person']

class Happiness(BaseScoringUDF):
    def __init__(self):
        super().__init__()
        self.arg_parser.add_argument("--class_thres", type=float, default=0.5)
        self.arg_parser.add_argument("--obj_thres", type=float, default=0)
        self.arg_parser.add_argument("--obj", type=str, choices=obj_names, default="person")
    
    def get_img_size(self):
        return (-1, -1)
    
    def get_scores(self, imgs, visualize=False):
        scores = []
        visual_imgs = []
        for img in imgs:
            obj=MultiFaceAnalyze.analyze(img)
            emotions = ["angry" , "disgust" , "fear" , "happy" , "sad" , "surprise" , "neutral"]
            weights = [-1, -1, -1, 1, -1, 1, 0]
            sum_happiness = 0
            print(obj)
            if visualize:
                visual_img = []
                visual_img = np.copy(img)
                #visual_img = cv2.cvtColor(visual_img, cv2.COLOR_BGR2RGB)
                factor_0 = 416 / visual_img.shape[0]
                factor_1 = 416 / visual_img.shape[1]
                factor = min(factor_0, factor_1)

                xx = int(visual_img.shape[1] * factor)
                yy = int(visual_img.shape[0] * factor)
                visual_img = cv2.resize(visual_img, (xx,yy))

            for face in obj.keys():
                face_data = obj[face]
                score = 0
                color = (255, 0, 0)
                for i in range(7):
                    emotion = emotions[i]
                    weight = weights[i]
                    score += face_data[emotion] * weight
                sum_happiness += score
                if visualize:
                    x1 = int(face_data['x'] / img.shape[1] * xx)
                    x2 = int((face_data['x'] + face_data['w']) / img.shape[1] * xx)
                    y1 = int(face_data['y'] / img.shape[0] * yy)
                    y2 = int((face_data['y'] + face_data['h']) / img.shape[0] * yy)
                    cv2.rectangle(visual_img, (x1, y1), (x2, y2), color, 2)
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = .6
                    thickness = 1
                    visual_img = cv2.putText(visual_img, face_data["dominant_emotion"], (x1, y2 + 20), font, fontScale, color, thickness, cv2.LINE_AA)
                    visual_img = cv2.putText(visual_img, 'score : {:.4f}'.format(score), (x1, y2 + 38), font, fontScale, color, thickness, cv2.LINE_AA)

            if len(obj.keys()) != 0:
                final_score = sum_happiness / len(obj.keys())
            else:
                final_score = 0
            scores.append(final_score)
            if visualize: 
                visual_imgs.append(Image.fromarray(visual_img))

        if visualize:
            return scores, visual_imgs
        else:
            return scores
