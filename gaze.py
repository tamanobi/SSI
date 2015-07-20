# coding:utf-8
import cv2, numpy as np
import dbutil, descriptor
import sys

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  names, kps = dbutil.LoadDB()
  flann = dbutil.IndexByFlann(kps)
  test = '../ccv_test/data/image{0:06d}.png'
  tar = int(sys.argv[1])
  img = cv2.imread(test.format(tar))
  voting = dbutil.QuerySearch(img, flann, len(names))
  kpslen = np.array([float(len(kp)) for kp in kps], dtype=np.float32)
  voting2 = voting.astype(np.float32) / kpslen
  rank = []
  for v,i in zip(voting.tolist(), [x for x in range(len(voting))]):
    rank.append([v,i])
  rank_sorted = sorted(rank, reverse=True)
  #plt.bar([x for x in range(len(voting2))],voting2)
  #plt.show()
  #idx = np.argmax(voting2)
  n_sim = 3
  idx = []
  for i in range(n_sim):
    #print rank_sorted[i][1]
    idx.append(rank_sorted[i][1])
  #print 'id:{0}, val:{1}, {2}, ratio{3}'.format(idx, np.max(voting2), len(kps[idx]), voting2[idx]*1.0/len(kps[idx]))
  for i,j in enumerate(idx):
    imgs = cv2.imread(test.format(j+1))
    cv2.imshow('rank{0:02d},{1:0.2e}'.format(i,rank_sorted[i][0]), imgs)
  cv2.imshow('query', img)
  cv2.waitKey(0)
