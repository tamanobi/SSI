# -*- coding: utf-8 -*-
import cv2
import numpy as np
import sys

def FASTandORB(img):
  row, col, channel = img.shape
  if channel is None:
    pass
  else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img = cv2.medianBlur(img, 3)
# FAST
  fast = cv2.FastFeatureDetector_create()
  kp_fast = fast.detect(img, None)
  #img_fast = cv2.drawKeypoints(img, kp_fast, None)
# ORB
  orb = cv2.ORB_create()
  kp_orb, des_orb = orb.compute(img, kp_fast)
  #img_des = cv2.drawKeypoints(img, kp_orb, None)
  #cv2.imshow('FAST&ORB', img_des)
  #cv2.imshow('FAST', img_fast)
  #cv2.waitKey(0)
  return des_orb

if __name__ == '__main__':
  argvs = sys.argv
  argc = len(argvs)
  print u'OpenCV Version:'+cv2.__version__
  print u'Python Version:'+str(sys.version_info)
  if argc == 2:
    img = cv2.imread(argvs[1])
    keypoints = FASTandORB(img)
    print 'n.features:{0}, n.dim:{1}'.format(keypoints.shape[0], keypoints.shape[1])
    print keypoints[0]
