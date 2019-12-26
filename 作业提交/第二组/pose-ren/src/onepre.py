import argparse
import os, sys
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)  # config
sys.path.append(os.path.join(ROOT_DIR, 'utils'))  # utils

import cv2
import numpy as np

from utils import util
from utils.model_pose_ren import ModelPoseREN
import numpy as np

from testing.net_deploy_baseline import make_baseline_net
from testing.net_deploy_pose_ren import make_pose_ren_net


def get_center(img, upper=650, lower=1):
    centers = np.array([0.0, 0.0, 300.0])
    count = 0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y, x] <= upper and img[y, x] >= lower:
                centers[0] += x
                centers[1] += y
                centers[2] += img[y, x]
                count += 1
    if count:
        centers /= count
    return centers
def load_name(namedir):
    with open(namedir) as f:
        return [line.strip() for line in f]


def load_center(centerdir):
    with open(centerdir) as f:
        return np.array([map(float,
            line.strip().split()) for line in f])


def pre(hand_model,base_dir,namedir,output):
    
    
    name=load_name(namedir)
    print name
    img=cv2.imread(base_dir+name[0],2)
    ct=get_center(img,500)
    print ct
    # center=load_center(centerdir)
    
    results = hand_model.detect_files(base_dir, name, np.array([ct]), max_batch=1)
    util.save_results(results, output+'output.txt')
    outputs = util.get_positions(output+'output.txt')
    img = img.astype(np.float32)
        # img[img >= 1000] = 1000
    img = (img - img.min())*255 / (img.max() - img.min())
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    resimg = util.draw_pose('icvl',img,outputs[0])
    print outputs[0]
    # cv2.imshow('res',resimg)
    # ch=cv2.waitKey(25)
    cv2.imwrite(output+'res.png',resimg)
    white = cv2.imread('/mnt/4224D24D24D24417/Datasets/hu-icvl/test/Depth/0/image.png')
    # img = white
    # img = np.uint8(img)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gujia = util.draw_pose('icvl',white,outputs[0])
    
    cv2.imwrite(output+'gujia.png',gujia)

if __name__ == "__main__":
    pre('/mnt/4224D24D24D24417/Datasets/hu-icvl/test/Depth/','./testname.txt')
