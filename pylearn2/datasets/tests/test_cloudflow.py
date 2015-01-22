from pylearn2.datasets.cloudflow import CLOUDFLOW
import math
import cv2
import pylab as plt
import numpy as np

import sys
model_file = sys.argv[1]

test = CLOUDFLOW(  
                 which_set='test',
                 num_examples=400000,
                 threshold=2,
                 prediv = 2,
                 postdiv = 2,
                 tdiv = 2,
                 train_frame_size = (3,25,25),
                 predict_frame_size = (2,1,1),
                 predict_interval = 2,
                 tstride = 1,
                 examples_per_image = 100,
                 video_shape = (7200, 477, 477),
                 image_border=(88, 88),
                 pad_border=(40, 40),
                 predict_style='interval',
                 track=True,
                 max_intensity=15.,
                 sampling_rates=(1., 1., 1., 1.),
                 rain_index_threshold=1.,
                 run_test=True,
                 model_file=model_file
                 )
#test.gen_random_examples2(test_mode=True)

def show_predictions(pred_stat):
    a = np.array(pred_stat)
    idx_arr = np.where(a[:,2] == 2)
    plt.plot(a[idx_arr, 0], a[idx_arr, 1], 'y,')
    for pred in pred_stat:
        if pred[2] == 0:
            plt.plot(pred[0], pred[1], 'g.')
        if pred[2] == 4:
            plt.plot(pred[0], pred[1], 'r.')
    plt.show()
"""
month, i, center, flow_mean  = test.show_random_examples(10000)
self = test
dt = self.train_frame_size[0] + self.predict_interval + self.predict_frame_size[0] - 1
dx = flow_mean[1] * dt * self.tdiv / self.prediv
dy = flow_mean[0] * dt * self.tdiv / self.prediv
dx = int(math.ceil(abs(dx)))
dy = int(math.ceil(abs(dy)))
dx
dy
diag_radius = int(math.ceil(math.sqrt(pow(self.train_frame_size[1]/2, 2) + 
                                    pow(self.train_frame_size[2]/2, 2))))
rmax = max(dx + diag_radius, dy + diag_radius)
rmax = int(math.ceil(rmax * math.sqrt(2.)))
radius = (rmax, rmax)
base_frames = self.get_frames(month, i, center, radius)
base_frames.shape

flow_mean = flow_mean.reshape((1,2))
mag, ang = cv2.cartToPolar(flow_mean[:,0], flow_mean[:,1], angleInDegrees=True)
angle = ang[0,0]
rot_mat = cv2.getRotationMatrix2D((radius[1], radius[0]), angle, 1.0)

j = 0
frame = base_frames[j]
rotated = cv2.warpAffine(frame, rot_mat, (frame.shape[1], frame.shape[0]))
dt_near = -(self.train_frame_size[0] + self.predict_interval - j)
dt_far = -(self.train_frame_size[0] + self.predict_interval + self.predict_frame_size[0] - 1 - j)
center0 = radius
flow = (mag[0,0], 0.)
center_near = self.translate_coords(center0, flow, dt_near)
center_far = self.translate_coords(center0, flow, dt_far)
r = self.train_frame_radius
cropped = rotated[center_near[0] - r[0] : center_near[0] + r[0], 
                center_far[1] - r[1] : center_near[1] + r[1]]
resized = cv2.resize(cropped, (cropped.shape[0], cropped.shape[0]))
"""

"""                 
trainset = CLOUDFLY(
                 which_set='train',
                 model_kind=3,
                 num_examples=560000,
                 threshold=3,
                 only_1Model = True,
                 onemodel_pixnum_threshold = 1,
                 prediv = 2,
                 postdiv = 2,
                 tdiv = 2,
                 train_frame_size = (3,30,30),
                 predict_frame_size = (1,1,1),
                 predict_interval = 2,
                 stride = (3,3),
                 tstride = 1,
                 tstart=0,
                 tstop=7200/2,
                 data_files = ['radar_img_matrix_AZ9280_201407_uint8.pkl.gz',
                               'radar_img_matrix_AZ9280_201408_uint8.pkl.gz',
                               'radar_img_matrix_AZ9280_201409_uint8.pkl.gz',
                               'radar_img_matrix_AZ9010_201406_uint8.pkl.gz',
                               'radar_img_matrix_AZ9010_201407_uint8.pkl.gz',
                               'radar_img_matrix_AZ9010_201408_uint8.pkl.gz',
                               'radar_img_matrix_AZ9010_201409_uint8.pkl.gz',
                               'radar_img_matrix_AZ9200_201406_uint8.pkl.gz',
                               'radar_img_matrix_AZ9200_201407_uint8.pkl.gz',
                               'radar_img_matrix_AZ9200_201408_uint8.pkl.gz',
                               'radar_img_matrix_AZ9200_201409_uint8.pkl.gz'],
                 examples_per_image = 100,
                 video_shape = (7200, 477, 477),
                 image_border=(90, 90))

valid = CLOUDFLY(
                 which_set='valid',
                 model_kind=3,
                 num_examples=220000,
                 threshold=3,
                 only_1Model = True,
                 onemodel_pixnum_threshold = 1,
                 prediv = 2,
                 postdiv = 2,
                 tdiv = 2,
                 train_frame_size = (3,30,30),
                 predict_frame_size = (1,1,1),
                 predict_interval = 2,
                 stride = (3,3),
                 tstride = 1,
                 tstart=0,
                 tstop=7200/2,
                 data_files = ['radar_img_matrix_AZ9280_201407_uint8.pkl.gz',
                               'radar_img_matrix_AZ9280_201408_uint8.pkl.gz',
                               'radar_img_matrix_AZ9280_201409_uint8.pkl.gz',
                               'radar_img_matrix_AZ9010_201406_uint8.pkl.gz',
                               'radar_img_matrix_AZ9010_201407_uint8.pkl.gz',
                               'radar_img_matrix_AZ9010_201408_uint8.pkl.gz',
                               'radar_img_matrix_AZ9010_201409_uint8.pkl.gz',
                               'radar_img_matrix_AZ9200_201406_uint8.pkl.gz',
                               'radar_img_matrix_AZ9200_201407_uint8.pkl.gz',
                               'radar_img_matrix_AZ9200_201408_uint8.pkl.gz',
                               'radar_img_matrix_AZ9200_201409_uint8.pkl.gz'],
                 examples_per_image = 100,
                 video_shape = (7200, 477, 477),
                 image_border=(90, 90))

test = CLOUDFLY(
                 which_set='test',
                 model_kind=3,
                 num_examples=340000,
                 threshold=3,
                 only_1Model = True,
                 onemodel_pixnum_threshold = 1,
                 prediv = 2,
                 postdiv = 2,
                 tdiv = 2,
                 train_frame_size = (3,30,30),
                 predict_frame_size = (1,1,1),
                 predict_interval = 2,
                 stride = (3,3),
                 tstride = 1,
                 tstart=0,
                 tstop=7200/2,
                 data_files = ['radar_img_matrix_AZ9280_201407_uint8.pkl.gz',
                               'radar_img_matrix_AZ9280_201408_uint8.pkl.gz',
                               'radar_img_matrix_AZ9280_201409_uint8.pkl.gz',
                               'radar_img_matrix_AZ9010_201406_uint8.pkl.gz',
                               'radar_img_matrix_AZ9010_201407_uint8.pkl.gz',
                               'radar_img_matrix_AZ9010_201408_uint8.pkl.gz',
                               'radar_img_matrix_AZ9010_201409_uint8.pkl.gz',
                               'radar_img_matrix_AZ9200_201406_uint8.pkl.gz',
                               'radar_img_matrix_AZ9200_201407_uint8.pkl.gz',
                               'radar_img_matrix_AZ9200_201408_uint8.pkl.gz',
                               'radar_img_matrix_AZ9200_201409_uint8.pkl.gz'],
                 examples_per_image = 100,
                 video_shape = (7200, 477, 477),
                 image_border=(90, 90))

testset = CLOUDFLY(
                 which_set='test',
                 model_kind=3,
                 num_examples=160000,
                 threshold=3,
                 only_1Model = True,
                 onemodel_pixnum_threshold = 1,
                 prediv = 2,
                 postdiv = 2,
                 tdiv = 2,
                 train_frame_size = (3,30,30),
                 predict_frame_size = (1,1,1),
                 predict_interval = 2,
                 stride = (10,10),
                 tstride = 1,
                 tstart=0,
                 tstop=7000/2*1,
                 data_files = ['radar_img_matrix_AZ9010_201409_uint8.pkl.gz',],
                 examples_per_image = None,
                 video_shape = (7000, 477, 477),
                 image_border=(90, 90))
"""