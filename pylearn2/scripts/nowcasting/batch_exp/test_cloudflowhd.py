from pylearn2.datasets.cloudflowhd import CloudFlowHD
from pylearn2.gui import patch_viewer

import math
import cv2
#import pylab as plt
import numpy as np

#import sys
#pre_ds = (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[2]))
#train_frame_size = (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[4]))
#pred_len = int(sys.argv[5])
#pred_interval = int(sys.argv[6])

#test = CloudFlowHD(
#                 which_set = 'test',
#                 examples_large = 400000,
#                 threshold = 3,
#                 tsmooth_win = 1,
#                 clip_shape = [6, 49, 49],
#                 ds_shape = [2, 4, 4],
#                 mean_tds = 0,
#                 pred_len = 20,
#                 interval_len = 5,
#                 examples_per_image = 5,
#                 sample_prob = 1.,
#                 test_mode = True
#    )

test = CloudFlowHD(
                 which_set = 'train',
                 examples_large = 100000,
                 threshold = 3,
                 tsmooth_win = 1,
                 clip_shape = [6, 17, 17],
                 ds_shape = [2, 1, 1],
                 mean_tds = 0,
                 pred_len = 20,
                 interval_len = 5,
                 tstride = 1,
                 examples_per_image = 800,
                 video_shape = (1800, 1066, 1000),
                 image_border=(40, 40),
                 train_slot=85,   # 17 hours
                 valid_slot=30,   # 6 hours
                 test_slot=40,   # 8 hours
                 sample_prob = 1.,
                 test_mode = True
    )
#test.test_flow()

def show_random_examples():
    example_cnt = test.X.shape[0]
    pv = patch_viewer.PatchViewer(grid_shape=(1, test.clip_shape_ds[0]), 
                                patch_shape=[test.clip_shape_ds[1], test.clip_shape_ds[2]])
    while True:
        i = np.random.randint(example_cnt)
        print 'i =', i
        print 'rain_bits =', test.y[i]
        print 'pred_bits =', test.y_pred[i]
        track_frames = test.X[i].reshape((3, 12, 12))
        print 'track_frames[-1].sum() =', track_frames[-1].sum()
        for j in range(track_frames.shape[0]):
            pv.add_patch(track_frames[j].astype('float32'), activation=0)
        pv.show()