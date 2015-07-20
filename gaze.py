# coding:utf-8
import cv2, numpy as np
import dbutil, descriptor
import sys

if __name__ == '__main__':
  argvs = sys.argv
  argc = len(argvs)
  if argc >= 3:
    num_similar = int(sys.argv[2])
  else:
    num_similar = 3
  if argc >= 2:
    # Loading Query Image
    query_file = sys.argv[1]
    img = cv2.imread(query_file)
    # Loading from database
    names, keypoints = dbutil.LoadDB()
    num_keypoints = np.array([float(len(kp)) for kp in keypoints], dtype=np.float32)
    # Indexing keypoints by flann(k-NN/kd-tree)
    flann = dbutil.IndexByFlann(keypoints)
    # Search simirality images from database 
    voting = dbutil.QuerySearch(img, flann, len(names))
    match_ratio = voting.astype(np.float32) / num_keypoints
    # Ranking based on match ratio
    rank = []
    for v,i in zip(match_ratio.tolist(), [x for x in range(len(voting))]):
      rank.append([v,i])
    rank_sorted = sorted(rank, reverse=True)
    idx = []
    for i in range(num_similar):
      idx.append(rank_sorted[i][1])
    # Display similarity images
    for i,j in enumerate(idx):
      imgs = cv2.imread(names[j])
      print 'rank:{0:02d}\tscore:{1:0.2e}\t{2}'.format(i+1,rank_sorted[i][0], names[j])
      cv2.imshow('rank{0:02d},{1:0.2e}'.format(i+1,rank_sorted[i][0]), imgs)
    cv2.imshow('query', img)
    cv2.waitKey(0)
  else:
    usage = '''
      usage: program [query filename] [num. of similarity images(opt/default=3)]
    '''
    print usage
