import os
import time

import cv2
import mmcv
import numpy as np
from PIL import Image

class ExecutionTimer(object):        
    def __init__(self, tag = 'NO NAME'):
        super()
        self.tag = tag
        
    def __enter__(self):
        self.wall_start = time.time()
        self.cpu_start = time.process_time()

    def __exit__(self, exception_type, exception_value, traceback):
        wt = time.time() - self.wall_start
        pt = time.process_time() - self.cpu_start
        print(f'{self.tag}: Execution Time is {wt} (wall) {pt} process')

root = 'D:/dataset/celeba/img_align_celeba'
files = os.listdir(root)
files = files[:200]

with ExecutionTimer("NP+PIL+float32"):
    for i in files:
        name = os.path.join(root, i)
        img = np.array(Image.open(name), np.float32)

with ExecutionTimer("NP+PIL+uint8"):
    for i in files:
        name = os.path.join(root, i)
        img = np.array(Image.open(name), np.uint8)

with ExecutionTimer("cv2"):
    for i in files:
        name = os.path.join(root, i)
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

with ExecutionTimer("mmcv"):
    for i in files:
        name = os.path.join(root, i)
        img = mmcv.imread(name)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
