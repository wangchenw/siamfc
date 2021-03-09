from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    seq_dir = os.path.expanduser('E:/Datasets/OTB100/hongwai/')
    img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
    anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt', delimiter=',')
    
    net_path = 'C:/Users/however/Desktop/siamfc-pytorch-master/pretrained/siamfc_alexnet_e49.pth'
    tracker = TrackerSiamFC(net_path=net_path)
    tracker.track(img_files, anno[0], visualize=True)
