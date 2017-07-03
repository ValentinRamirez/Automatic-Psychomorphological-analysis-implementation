import facial_feature_detector as feature_detection
import cv2
import numpy as np
import os
import check_resources as check
import matplotlib.pyplot as plt
from PM_analysis import Three_zones,Triangle_of_senses,Exp_Ret
import math


this_path = os.path.dirname(os.path.abspath(__file__))

def main():
  img = cv2.imread("Caras/einstein.jpg", 1)
  plt.title('Query Image')
  plt.imshow(img[:, :, ::-1])
  # check for dlib saved weights for face landmark detection
  # if it fails, dowload and extract it manually from
  # http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
  check.check_dlib_landmark_weights()
  # extract landmarks from the query image
  # list containing a 2D array with points (x, y) for each face detected in the query image
  lmarks = feature_detection.get_landmarks(img)
  plt.figure()
  plt.title('Landmarks Detected')
  plt.imshow(img[:, :, ::-1])
  plt.scatter(lmarks[0][:, 0], lmarks[0][:, 1])
#Three_zones
  rational_area, emotional_area, instinctive_area, chin_area = Three_zones(img, lmarks)
  BF_area=rational_area+emotional_area+instinctive_area
  print '\nRational area=',rational_area,'pixels','-',100*rational_area/BF_area,'%\nEmotional area=',emotional_area,'pixels','-',100*emotional_area/BF_area ,'%\nInstinctive area=',instinctive_area,'pixels','-', 100*instinctive_area/BF_area,'%'

#Triangle of senses
  Tr_ratio=Triangle_of_senses(img,BF_area,lmarks)
  print '\nTriangle of Senses ratio', Tr_ratio
#Expansion/Retraccion
  ER_ratio=Exp_Ret(img,chin_area,lmarks)
  print '\nExpansion/Retraction=', ER_ratio




if __name__ == "__main__":
      main()
