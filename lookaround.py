# -*- coding: utf-8 -*-
import sys, glob
import numpy as np
import cv2

import descriptor, dbutil

if __name__ == '__main__':
  argvs = sys.argv
  argc = len(argvs)
  if argc == 2:
    dir_path = argvs[1]
    file_list= glob.glob(dir_path+'*')
    dbutil.ResetDB()
    for i,fname in enumerate(file_list):
      print i,fname
      im = cv2.imread(fname)
      keypoints = descriptor.FASTandORB(im)
      dbutil.RegistFilenames(i, fname)
      dbutil.RegistKeypoints(i, keypoints)
