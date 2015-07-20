# -*- coding:utf-8 -*-
import codecs
import descriptor
def ResetDB():
  codecs.open('names.csv', 'w', 'utf-8')
  codecs.open('keypoints.csv', 'w', 'utf-8')

def LoadDB():
  names = []
  kps = []
  with codecs.open('names.csv', 'r', 'utf-8') as f:
    while True:
      line = f.readline()
      line = line.rstrip()
      if (len(line)==0) or (line is None):
        break
      record = line.split('\t', 1)
      names.append(record[1])
  with codecs.open('keypoints.csv', 'r', 'utf-8') as f:
    prev_id = 0
    des = []
    while True:
      line = f.readline()
      line = line.rstrip()
      if (len(line)==0) or (line is None):
        kps.append(des)
        break
      record = line.split('\t')
      if prev_id != int(record[0]):
        prev_id = int(record[0])
        kps.append(des)
        des = []
      des.append(map(int, record[1:]))
  if len(kps) == 0:
    print 'keypoints is empty'
  if len(names) == 0:
    print 'names is empty'
  return names, kps

def IndexByFlann(kps):
  import numpy as np
  import cv2
  FLANN_INDEX_KDTREE = 0
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
  search_params = dict(checks=10)
  flann = cv2.FlannBasedMatcher(index_params,search_params)
  for kp in kps:
    flann.add([np.array(kp, dtype=np.float32)])
  return flann

def QuerySearch(img, flann, n):
  import numpy as np
  #orb = cv2.ORB_create()
  #kp1, des1 = orb.detectAndCompute(img, None)
  des1 = descriptor.FASTandORB(img)
  des1 = des1.astype(np.float32)
  matches = flann.knnMatch(des1,k=2)
  voting = np.zeros(n)
  for match in matches:
    for m in match:
      voting[m.imgIdx] += 1
  return voting

def RegistFilenames(num, filename):
  with codecs.open('names.csv', 'a+', 'utf-8') as f:
    writebuffer = '{0}\t{1}'.format(num, filename)+'\n'
    f.write(writebuffer)

def RegistKeypoints(num, keypoints):
  with codecs.open('keypoints.csv', 'a+', 'utf-8') as f:
    for kp in keypoints:
      writebuffer = '{0}\t'.format(num) +\
      '\t'.join(map(str, kp))+'\n'
      f.write(writebuffer)

def LoadDB_check():
  names, kps = LoadDB()
  print names[0]
  print kps[0][1],len(kps[0][1])


if __name__ == '__main__':
  pass
