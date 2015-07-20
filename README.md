### files ###
* dbutil.py: database(plain tsv) utilities
* lookaround.py: creating database
* descriptor.py: extracting feature(FAST/ORB)
* gaze.py: searching similarity images interface

### usage1:database ###
  python lookaround.py [directory path]

* directory path: directory's path contains only images(excluding gif)

### usage2:searching ###
  python gaze.py [filename] [num_sim]

* filename: query file name
* num_sim: number of displaying similarity images(default=3)
