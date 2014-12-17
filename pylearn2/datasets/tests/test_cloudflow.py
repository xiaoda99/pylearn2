from pylearn2.datasets.cloudflow import CLOUDFLY, CLOUDFLOW, CLOUDFLOW2
import math
import cv2
import pylab as plt

test = CLOUDFLOW2(  
                 which_set='train',
                 num_examples=620000,
                 threshold=3,
                 pixnum_threshold = 1,
                 prediv = 2,
                 postdiv = 2,
                 tdiv = 2,
                 train_frame_size = (3,24,24),
                 predict_frame_size = (1,1,1),
                 predict_interval = 2,
                 stride = (3,3),
                 tstride = 1,
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
                 image_border=(88, 88),
                 pad_border=(40, 40),
                 train_slot=50,   # 5 hours
                 valid_slot=20,   # 2 hours
                 test_slot=30,   # 3 hours
                 predict_style='point',
                 )

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
frame = base_frames[-1]
plt.imshow(frame); plt.show()

flow_mean = flow_mean.reshape((1,2))
mag, ang = cv2.cartToPolar(flow_mean[:,0], flow_mean[:,1], angleInDegrees=True)
angle = ang[0,0]
angle
rot_mat = cv2.getRotationMatrix2D((radius[1], radius[0]), angle, 1.0)
rotated = cv2.warpAffine(frame, rot_mat, (frame.shape[1], frame.shape[0]))
rotated.shape
plt.imshow(rotated); plt.show()
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