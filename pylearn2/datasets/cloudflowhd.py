import sys
from time import time
import os, cPickle, logging
import traceback
_logger = logging.getLogger(__name__)

import numpy
np = numpy
import gzip
import cv2
import cv2.cv as cv
import random
import math
from pprint import pprint

from theano import function
from pylearn2.datasets import dense_design_matrix
from pylearn2.gui import patch_viewer
from pylearn2.utils import serial

default_data_files = [
                      'radar_img_matrix_AZ9280_201407_uint8.pkl.gz',
#                       'radar_img_matrix_AZ9280_201408_uint8.pkl.gz',
#                       'radar_img_matrix_AZ9280_201409_uint8.pkl.gz',
#                       'radar_img_matrix_AZ9010_201406_uint8.pkl.gz',
#                       'radar_img_matrix_AZ9010_201407_uint8.pkl.gz',
#                       'radar_img_matrix_AZ9010_201408_uint8.pkl.gz',
#                       'radar_img_matrix_AZ9010_201409_uint8.pkl.gz',
#                       'radar_img_matrix_AZ9200_201406_uint8.pkl.gz',
#                       'radar_img_matrix_AZ9200_201407_uint8.pkl.gz',
#                       'radar_img_matrix_AZ9200_201408_uint8.pkl.gz',
#                       'radar_img_matrix_AZ9200_201409_uint8.pkl.gz',
                       ]

def downsample(frames, ds_shape, dtype='float32', rescale=1., mean_tds=True):
    frames_ds = np.zeros((frames.shape[0],
                          frames.shape[1]/ds_shape[1],
                          frames.shape[2]/ds_shape[2]), 
                         dtype='float32')
    for i in range(frames.shape[0]):
        frames_ds[i] = cv2.resize(frames[i].astype('float32'), (0, 0), fx=1./ds_shape[1], fy=1./ds_shape[2],
                                  interpolation = cv2.INTER_AREA)
    if ds_shape[0] > 1:
        if mean_tds: 
            frames_ds = frames_ds.reshape((frames_ds.shape[0]/ds_shape[0], ds_shape[0], 
                        frames_ds.shape[1], frames_ds.shape[2])).mean(axis=1)
        else:
            frames_ds = frames_ds[(frames_ds.shape[0]-1) % ds_shape[0] : : ds_shape[0]]
    frames_ds *= rescale
    assert dtype in ['float32', 'uint8']
    if dtype == 'uint8':
        frames_ds = frames_ds.round().astype('uint8')
    return frames_ds

def downsample_flow(frames, ds_shape, dtype='float32', rescale=1.):
    frames_ds = np.zeros((frames.shape[0],
                          frames.shape[1]/ds[1],
                          frames.shape[2]/ds[2],
                          2), 
                         dtype='float32')
    for i in range(frames.shape[0]):
        frames_ds[i] = cv2.resize(frames[i].astype('float32'), (0, 0), fx=1./ds[1], fy=1./ds[2])
    if ds[0] > 1:
        frames_ds = frames_ds.reshape((frames_ds.shape[0]/ds[0], ds[0], 
                        frames_ds.shape[1], frames_ds.shape[2], 2)).mean(axis=1)
    frames_ds *= rescale
    assert dtype in ['float32', 'int8']
    if dtype == 'int8':
        frames_ds = frames_ds.round().astype('int8')
    return frames_ds

from profilehooks import profile

class Predictor():
    def __init__(self, pred_len, interval_len):
        self.__dict__.update(locals())
        del self.self
        self.n_intervals = self.pred_len / self.interval_len
        
        self.model_paths = [
#                       '4out_cnn_mtds0_i0_best.pkl',
#                       '4out_cnn_mtds0_i1_best.pkl',
#                       '4out_cnn_mtds0_i2_best.pkl',
#                       '4out_cnn_mtds0_i3_best.pkl',
                                           
                        '4out_clip[6,49,49]_ds[2,4,4]_mtds0_nvis432_best.pkl',
                        '4out_clip[6,49,49]_ds[2,4,4]_mtds0_nvis432_best.pkl',
                        '4out_clip[6,49,49]_ds[2,4,4]_mtds0_nvis432_best.pkl',
                        '4out_clip[6,49,49]_ds[2,4,4]_mtds0_nvis432_best.pkl',
                        ]
        
        self.pred_fns = [self._compile_pred_fn(self.model_paths[i]) for i in range(self.n_intervals)]
        
        self.ds_shape = []
        self.mean_tds = []
        for i in range(self.n_intervals):
            self.ds_shape.append(self._parse_model_path(self.model_paths[i])['ds_shape'])
            self.mean_tds.append(self._parse_model_path(self.model_paths[i])['mean_tds'])
    
    def _compile_pred_fn(self, model_path):
        model = serial.load(model_path)
        print 'compiling pred_fn for model', model_path, '...'
        X = model.get_input_space().make_batch_theano()
        y = model.fprop(X)
        fn = function([X], y,allow_input_downcast=True)
        self.topo_view = True if X.ndim == 4 else False
        print 'done.'
        return fn
    
    def _parse_model_path(self, model_path):
        # model_path example: 4out_clip[6,49,49]_ds[2,4,4]_mtds0_nvis432_i2_best.pkl
#        params = model_path.split('_')
#        ds_str = params[2].replace('ds[', '').replace(']', '')
#        ds = [int(s) for s in ds_str.split(',')]
#        
#        mtds_str = params[3].replace('mtds', '')
#        mtds = int(mtds)
#        
#        return {'ds_shape' : ds, 'mean_tds' : mtds}
        return {'ds_shape' : [2,4,4], 'mean_tds' : 0}
    
    @profile
    def predict(self, clipper, center):
        trace = clipper.flow_back(center, self.pred_len + clipper.clip_shape[0] - 1, radius=clipper.clip_radius)
        if trace is None:
            return [0.] * self.n_intervals, None
        
        rain_probs = []
        for i in range(self.n_intervals):
            track_frames = clipper.clip_track_frames(trace, self.interval_len * i + 1, self.interval_len * (i+1))
            if np.prod(self.ds_shape[i]) > 1:
                track_frames = downsample(track_frames, self.ds_shape[i], dtype='uint8', rescale=10., 
                                          mean_tds=self.mean_tds[i])
            
            if not self.topo_view:
                x = track_frames.reshape(1, track_frames.size) # for MLP
            else:
                x = track_frames.reshape(track_frames.shape + (1,)) # for CNN, axis = ('c', 0, 1, 'b')
            rain_prob = self.pred_fns[i](x)[0, i]
            rain_probs.append(rain_prob)
        
        return rain_probs, trace
    
class Monitor():
    def __init__(self, pred_len, interval_len):
        self.__dict__.update(locals())
        del self.self
        self.n_intervals = self.pred_len / self.interval_len
        
        self.rain = np.zeros(self.n_intervals, dtype='int')
        self.pred_baseline = np.zeros(self.n_intervals, dtype='int')
        self.tp_baseline = np.zeros(self.n_intervals, dtype='int')
        self.pred = np.zeros(self.n_intervals, dtype='int')
        self.tp = np.zeros(self.n_intervals, dtype='int')
    
    def check_prediction(self, rain_bits, pred_bits_baseline, rain_probs):
        rain_probs = np.array(rain_probs)
        pred_bits = rain_probs >= 0.5
        
        self.rain += rain_bits
        self.pred_baseline += pred_bits_baseline
        self.tp_baseline += rain_bits * pred_bits_baseline
        self.pred += pred_bits
        self.tp += rain_bits * pred_bits
        
    def print_results(self):
        precision_baseline = self.tp_baseline * 1. / self.pred_baseline
        recall_baseline = self.tp_baseline * 1. / self.rain
        f1_baseline = precision_baseline * recall_baseline * 2. / (precision_baseline + recall_baseline)
    
        precision = self.tp * 1. / self.pred
        recall = self.tp * 1. / self.rain
        f1 = precision * recall * 2. / (precision + recall)
        
        print 'Baseline'
        print precision_baseline, ':precision'
        print recall_baseline, ':recall'
        print f1_baseline, ':f1'
        print 'Tracknn'
        print precision, ':precision'
        print recall, ':recall'
        print f1, ':f1'
        
class Clipper():
    def __init__(self,
                 matrix,
                 flow,
                 image_border,
                 clip_shape):
        self.__dict__.update(locals())
        del self.self
        assert (self.clip_shape[1] - 1) % 2 == 0 and (self.clip_shape[2] - 1) % 2 == 0
        self.clip_radius = ((self.clip_shape[1] - 1) / 2, (self.clip_shape[2] - 1) / 2)
        diag_radius = int(math.ceil(math.sqrt(pow(self.clip_radius[0], 2) + pow(self.clip_radius[1], 2))))
        self.diag_radius = (diag_radius, diag_radius)
    
    def _get_flow(self, center, radius=(0,0)):
        row, col = center
        if radius == (0,0):
            flow = self.flow[-1, row, col].astype('float') / 10.
            return flow[1], flow[0]
        else:
            flow = self.flow[-1, row - radius[0] : row + radius[0] + 1, 
                             col - radius[1] : col + radius[1] + 1].astype('float').mean(axis=(0,1)) / 10.
            return flow[1], flow[0]
        
    def _inbounds(self, center, radius, logical=True):
        if logical:
            return center[0] >= self.image_border[0] + radius[0] and \
                center[0] < self.matrix.shape[1] - self.image_border[0] - radius[0] and \
                center[1] >= self.image_border[1] + radius[1] and \
                center[1] < self.matrix.shape[2] - self.image_border[1] - radius[1]
        else:
            return center[0] >= radius[0] and \
                center[0] < self.matrix.shape[1] - radius[0] and \
                center[1] >= radius[1] and \
                center[1] < self.matrix.shape[2] - radius[1]
        
    def flow_back(self, origin, steps, const_flow=None, radius=(0,0)):
        row, col = origin
        row = float(row); col = float(col) # internal representation of row and col must be float
                                           # to avoid rounding error accumulation
        trace = [origin,]
        for i in range(steps):
            if const_flow is not None:
                drow, dcol = const_flow
            else:
                drow, dcol = self._get_flow((int(round(row)), int(round(col))), radius)
            row -= drow; col -= dcol
            if not self._inbounds((int(round(row)), int(round(col))), self.diag_radius, logical=True): 
                return None
            trace.append((int(round(row)), int(round(col))))
        return trace
    
    def _get_frame(self, frame_idx, center, radius):
        return self.matrix[frame_idx, 
                    center[0] - radius[0] : center[0] + radius[0] + 1,
                    center[1] - radius[1] : center[1] + radius[1] + 1]
    
    def _get_base_radius(self, center_near, center_far):
        dcenter = max(abs(center_near[0] - center_far[0]), abs(center_near[1] - center_far[1]))
        base_radius = int(math.ceil((dcenter + self.diag_radius[0]) * math.sqrt(2.)))
        return (base_radius, base_radius) 
        
    def _clip_track_frame(self, frame_idx, center_near, center_far):
        radius = self._get_base_radius(center_near, center_far)
        assert self._inbounds(center_near, radius, logical=False)
        base_frame = self._get_frame(frame_idx, center_near, radius)
        
        dx = np.array([center_near[1] - center_far[1]])
        dy = np.array([center_near[0] - center_far[0]])
        mag, ang = cv2.cartToPolar(dx.astype('float32'), dy.astype('float32'), angleInDegrees=True)
        mag = mag[0,0]
        ang = ang[0,0]
        rot_mat = cv2.getRotationMatrix2D((radius[1], radius[0]), ang, 1.0)
        rotated = cv2.warpAffine(base_frame, rot_mat, (base_frame.shape[1], base_frame.shape[0]))
        
        # compute new coordinates relative to base_frame
        center_near = radius
        center_far = (radius[0], int(round(radius[1] - mag)))
        
        r = ((self.clip_shape[1] - 1) / 2, (self.clip_shape[2] - 1) / 2)
        cropped = rotated[center_near[0] - r[0] : center_near[0] + r[0] + 1, 
                            center_far[1] - r[1] : center_near[1] + r[1] + 1]
        track_frame = cv2.resize(cropped, (cropped.shape[0], cropped.shape[0]), interpolation = cv2.INTER_AREA)
        return track_frame
    
    def clip_track_frames(self, trace, trace_idx_near, trace_idx_far):
        frames = np.zeros(self.clip_shape)
        for i in range(self.clip_shape[0]):
            center_near = trace[trace_idx_near + self.clip_shape[0] - 1 - i]
            center_far = trace[trace_idx_far + self.clip_shape[0] - 1 - i]
            frame = self._clip_track_frame(i, center_near, center_far)
            frames[i] = frame  # reverse time order from present->past to past->present
        return frames
    
class CloudFlowHD(dense_design_matrix.DenseDesignMatrix):
    matrix = None
    flow = None
    
    def __init__(self,  
                 which_set,
                 examples_large,
                 threshold = 3,
                 tsmooth_win = 1,
                 preds_shape = (1, 1, 1),
                 clip_shape = (6,61,61),
                 ds_shape = (1, 1, 1),
                 mean_tds = True,
                 pred_len = 20,
                 interval_len = 5,
                 pred_interval_idx = None,
                 tstride = 1,
                 data_files = default_data_files,
                 axes=('c', 0, 1, 'b'),
                 examples_per_image = None,
                 video_shape = (7200, 477, 477),
                 image_border=(88, 88),
                 train_slot=100,   # 10 hours
                 valid_slot=40,   # 4 hours
                 test_slot=50,   # 5 hours
                 sample_prob=1.,
                 reverse_flow=False,
                 test_mode=False
                 ):
            
        self.__dict__.update(locals())
        del self.self
        assert self.clip_shape[1] % 2 == 1 and self.clip_shape[2] % 2 == 1
        self.clip_radius = ((self.clip_shape[1]-1)/2, (self.clip_shape[2]-1)/2)
        self.n_intervals = self.pred_len / self.interval_len
        self.tsmooth_wins = [1, 2, 3]
        
        dummy_frames = np.zeros(self.clip_shape, dtype='float32')
        dummy_frames = downsample(dummy_frames, self.ds_shape, dtype='uint8', rescale=10.)
        self.clip_shape_ds = dummy_frames.shape
        print 'self.clip_shape_ds =', self.clip_shape_ds
        
        self.init_slots()
        
        
#        self.flow_pred_stat = self.create_pred_stat()
#        self.flow_pred_stat0 = self.create_pred_stat()
#        self.flow_pred_stat1 = self.create_pred_stat()
#        self.flow_pred_stat2 = self.create_pred_stat()
        
        ramdisk_root = '/home/xd/ramdisk_backup/'
        X_path = ramdisk_root + self.filename('X')
        y_path = ramdisk_root + self.filename('y')
         
        if not self.test_mode and os.path.isfile(X_path):
            print '\n', which_set, 'set already built. Loading from file...'
            self.X_large = np.load(X_path)
            self.y_large = np.load(y_path)
            self.example_cnt = self.X_large.shape[0]
            print 'Done.'
        else:
            if self.test_mode:
                self.predictor = Predictor(self.pred_len, self.interval_len)
                self.monitor = Monitor(self.pred_len, self.interval_len)  # monitor for the whole dataset
            else:
                self.X_large = np.zeros((self.examples_large, np.prod(self.clip_shape_ds)), dtype='uint8')
                self.y_large = np.zeros((self.examples_large, self.n_intervals * len(self.tsmooth_wins) * 2), dtype='bool_')
                self.example_cnt = 0
            
            if CloudFlowHD.matrix is None:
                CloudFlowHD.matrix = np.zeros(self.video_shape, dtype='uint8') 
                CloudFlowHD.flow = np.zeros(self.video_shape + (2,), dtype='int8')
                
            for month in range(len(data_files)):
                data_file = data_files[month]
                matrix_file = ramdisk_root + data_file.replace('.pkl.gz', '_hd.npy')
                if not self.reverse_flow:
                    flow_file = ramdisk_root + data_file.replace('.pkl.gz', '_flow_hd.npy')
                else:
                    flow_file = ramdisk_root + data_file.replace('.pkl.gz', '_flow_rev_hd.npy')
                if not os.path.isfile(matrix_file):
                    print 'Building matrix for', data_file, '...'
                    t0 = time()
                    matrix = self.build_matrix(data_file)
                    print 'Done.', time() - t0, 'seconds'
                    
                    print 'Building flow for', data_file, '...'
                    t0 = time()
                    flow = self.build_flow(matrix, reverse=False)
                    print 'Done.', time() - t0, 'seconds'
                    
                    print 'Building reverse flow for', data_file, '...'
                    t0 = time()
                    flow_rev = self.build_flow(matrix, reverse=True)
                    print 'Done.', time() - t0, 'seconds'
                    
                    print 'Caching data for', data_file, 'to ramdisk...'
                    t0 = time()
                    np.save(matrix_file, matrix)
                    np.save(flow_file, flow)
                    np.save(flow_rev_file, flow_rev)
                    print 'Done.', time() - t0, 'seconds'
                    
                print 'Loading data for', data_file, 'from ramdisk...'
                t0 = time()
                CloudFlowHD.matrix[:,:,:] = np.load(matrix_file)
                CloudFlowHD.flow[:,:,:,:] = np.load(flow_file)
    #            self.matrix = matrix; self.flow = flow
                print 'Done.', time() - t0, 'seconds'
                
                assert np.prod(self.preds_shape) == 1
                if np.prod(self.preds_shape) > 1:
                    print 'Downsampling...'
                    t0 = time()
                    matrix = downsample(matrix, self.preds_shape, dtype='uint8')
                    self.image_border = (np.ceil(self.image_border[0]/self.preds_shape[1]), 
                                         np.ceil(self.image_border[1]/self.preds_shape[2]))
                    self.train_slot /= self.preds_shape[0]
                    self.valid_slot /= self.preds_shape[0]
                    self.test_slot /= self.preds_shape[0]
                    flow = downsample_flow(flow, self.preds_shape, dtype='int8', 
                                                rescale=self.preds_shape[0]*1./self.preds_shape[1])
                    print 'Done.', time() - t0, 'seconds'
                
                self.matrix = CloudFlowHD.matrix
                self.flow = CloudFlowHD.flow
                if not self.test_mode:
                    self.gen_random_examples()
                else:
                    self.test_random_examples()
#                self.example_cnt = self.examples_large - 100
#                del matrix
#                del flow
                
            if self.test_mode:
                self.monitor.print_results()
                return
            
            print 'Saving', which_set, 'set to file...'
            self.X_large = self.X_large[:self.example_cnt]
            self.y_large = self.y_large[:self.example_cnt]
            np.save(X_path, self.X_large)
            np.save(y_path, self.y_large)
            print 'Done.'
        
        X, y, y_pred = self.sample(self.X_large, self.y_large)
        del self.X_large
        del self.y_large
        
        if self.pred_interval_idx is not None:
            mask = np.zeros(self.n_intervals, dtype='bool_')
            mask[self.pred_interval_idx] = 1
            y *= mask
            y_pred *= mask
        
        shape = (self.clip_shape_ds[1],  #rows
                 self.clip_shape_ds[2],  #cols
                 self.clip_shape_ds[0]   #frames, i.e. channels
                 )     
        view_converter = dense_design_matrix.DefaultViewConverter(shape, self.axes)
        
        super(CloudFlowHD, self).__init__(X = X, y = y, view_converter = view_converter)
        self.y_pred = y_pred
                
#        print 'exact flow'
#        self.calc_pred_stat(self.flow_pred_stat)
#            print 'exact mean flow'
#            self.calc_pred_stat(self.flow_pred_stat0)
#        print 'exact mean flow'
#        self.calc_pred_stat(self.flow_pred_stat1)
#        print 'approximate mean flow'
#        self.calc_pred_stat(self.flow_pred_stat2)
    
    def filename(self, X_or_y):
        s = self.which_set
        s += '_th' + str(self.threshold)
        s += '_' + str(self.clip_shape[0]) + 'x' + \
                    str(self.clip_shape[1]) + 'x' + \
                    str(self.clip_shape[2])
        s += '-ds' + str(self.ds_shape[0]) + 'x' + \
                    str(self.ds_shape[1]) + 'x' + \
                    str(self.ds_shape[2])
        s += '-' + str(self.interval_len) + 'x' + str(self.n_intervals)
        s += '_epi' + str(self.examples_per_image)
        s += '_mean_tds' + str(int(self.mean_tds))
        s += ('_' + X_or_y +'_uint8x10.npy')
        return s
    
    def sample(self, X_large, y_large):
        example_cnt = X_large.shape[0]
        sampled = np.random.uniform(0., 1., example_cnt) < self.sample_prob
        sampled = sampled | (self.which_set == 'test') | y_large.max(axis=1)
        print 'example_cnt =', example_cnt 
        print 'sampled_cnt =', sampled.sum()
        X = X_large[np.where(sampled == 1)]
        y = self._load_rain_bits_batch(y_large, self.tsmooth_win)[np.where(sampled == 1)]
        y_pred = self._load_pred_bits_batch(y_large, self.tsmooth_win)[np.where(sampled == 1)]  
        return X, y, y_pred
    
    def init_slots(self):
        self.whole_slot = self.train_slot + self.valid_slot + self.test_slot
        assert self.which_set in ['train', 'valid', 'test']
        if self.which_set == 'train':
            self.usable_start = 0
            self.usable_stop = self.train_slot - 1
        elif self.which_set == 'valid':
            self.usable_start = self.train_slot
            self.usable_stop = self.train_slot + self.valid_slot - 1
        else:
            self.usable_start = self.train_slot + self.valid_slot
            self.usable_stop = self.train_slot + self.valid_slot + self.test_slot - 1
                          
    def usable(self, pos):
        start = (pos - self.clip_shape[0] + 1) % self.whole_slot
        stop = start + self.clip_shape[0] - 1 + self.pred_len
        return start >= self.usable_start and stop <= self.usable_stop
    
    def get_random_center(self):
        row = np.random.randint(self.image_border[0] + self.clip_radius[0], 
                                self.matrix.shape[1] - self.image_border[0] - self.clip_radius[0])
        col = np.random.randint(self.image_border[1] + self.clip_radius[1], 
                                self.matrix.shape[2] - self.image_border[1] - self.clip_radius[1])
        center = (row, col)
        return center
                
    def detect_rain(self, intensities, tsmooth_win):
        if tsmooth_win > 1:
            tsmooth_kernel = np.ones((tsmooth_win,))/tsmooth_win
            intensities = np.convolve(intensities, tsmooth_kernel)[(tsmooth_win-1) : -(tsmooth_win-1)]
        return intensities.max() >= self.threshold
    
    def _get_rain_bits(self, i, trace, tsmooth_win=1):
        rain_bits = np.zeros(self.n_intervals, dtype='bool_')  
        row, col = trace[0]
        for j in range(self.n_intervals):
            intensities = self.matrix[i + 1 + self.n_intervals * j : i + 1 + self.n_intervals * (j + 1), row, col]
            rain_bits[j] = self.detect_rain(intensities, tsmooth_win) 
        return rain_bits
    
    def _get_rain_bits_track(self, i, trace, tsmooth_win):
        rain_bits = np.zeros(self.n_intervals, dtype='bool_')    
        for j in range(self.n_intervals):
            trace_idx = self.pred_len - self.interval_len - self.interval_len * j
            interval_start = i + 1 + self.interval_len * j
            interval_stop = i + 1 + self.interval_len + self.interval_len * j
            intensities = self.matrix[interval_start : interval_stop, trace[trace_idx][0], trace[trace_idx][1]]
            rain_bits[j] = self.detect_rain(intensities, tsmooth_win) 
        return rain_bits
    
    def _get_pred_bits(self, i, trace, tsmooth_win=1):
        pred_bits = np.zeros(self.n_intervals, dtype='bool_')    
        for j in range(self.n_intervals):
            trace_seg = trace[self.interval_len * j + 1 : self.interval_len * (j+1) + 1]
            rows = [p[0] for p in trace_seg]
            cols = [p[1] for p in trace_seg]
            intensities = self.matrix[i, rows, cols]
            pred_bits[j] = self.detect_rain(intensities, tsmooth_win) # flow predictions for all intervals are the same
        return pred_bits
    
    def _get_pred_bits_track(self, i, trace, tsmooth_win):
        pred_bits = np.zeros(self.n_intervals, dtype='bool_')    
        trace_seg = trace[self.pred_len - self.interval_len + 1 : self.pred_len + 1]
        rows = [p[0] for p in trace_seg]
        cols = [p[1] for p in trace_seg]
        intensities = self.matrix[i, rows, cols]
        pred_bits[:] = self.detect_rain(intensities, tsmooth_win) # flow predictions for all intervals are the same
        return pred_bits
    
    def _save_rain_bits(self, bits_arr, tsmooth_win, rain_bits):
        bits_arr[(tsmooth_win - 1) * self.n_intervals : tsmooth_win * self.n_intervals] = rain_bits
        
    def _load_rain_bits(self, bits_arr, tsmooth_win):
        rain_bits = bits_arr[(tsmooth_win - 1) * self.n_intervals : tsmooth_win * self.n_intervals]
        return rain_bits
    
    def _load_rain_bits_batch(self, bits_mat, tsmooth_win):
        rain_bits_batch = bits_mat[:, (tsmooth_win - 1) * self.n_intervals : tsmooth_win * self.n_intervals]
        return rain_bits_batch
    
    def _save_pred_bits(self, bits_arr, tsmooth_win, pred_bits):
        bits_arr[self.n_intervals * len(self.tsmooth_wins) + (tsmooth_win - 1) * self.n_intervals : 
                 self.n_intervals * len(self.tsmooth_wins) + tsmooth_win * self.n_intervals] = pred_bits
                 
    def _load_pred_bits(self, bits_arr, tsmooth_win):
        pred_bits = bits_arr[self.n_intervals * len(self.tsmooth_wins) + (tsmooth_win - 1) * self.n_intervals : 
                 self.n_intervals * len(self.tsmooth_wins) + tsmooth_win * self.n_intervals]
        return pred_bits
    
    def _load_pred_bits_batch(self, bits_mat, tsmooth_win):
        pred_bits_batch = bits_mat[:, self.n_intervals * len(self.tsmooth_wins) + (tsmooth_win - 1) * self.n_intervals : 
                 self.n_intervals * len(self.tsmooth_wins) + tsmooth_win * self.n_intervals]
        return pred_bits_batch
    
    def is_empty(self, frames):
        return frames[-1].sum() == 0
    
    def gen_random_examples(self):
        print 'Generating random examples ...'
        t0 = time()
        
        total = 0; usable = 0; example = 0
        for i in range(self.clip_shape[0], self.matrix.shape[0] - self.pred_len):
            total += 1
            if not self.usable(i):
                continue
            usable += 1
            clipper = Clipper(self.matrix[i + 1 -self.clip_shape[0] : i + 1], 
                              self.flow[i + 1 -self.clip_shape[0] : i + 1], 
                              self.image_border, self.clip_shape)
            for _ in range(self.examples_per_image):
                center = self.get_random_center()
                trace = clipper.flow_back(center, self.pred_len + self.clip_shape[0] - 1, radius=self.clip_radius)
                if trace is None:
                    continue    
                
                track_frames = clipper.clip_track_frames(trace, self.pred_len - self.interval_len + 1, self.pred_len)
                 
                if np.prod(self.ds_shape) > 1:
                    track_frames = downsample(track_frames, self.ds_shape, dtype='uint8', rescale=10., 
                                              mean_tds=self.mean_tds)
                    
                if self.which_set != 'test' and self.is_empty(track_frames):   
#                if self.is_empty(track_frames):
                    continue
                
                self.X_large[self.example_cnt] = track_frames.flatten()
                
                # targets for different temporal smoothing windows
                bits_arr = np.zeros(self.n_intervals * len(self.tsmooth_wins) * 2, dtype='bool_') # for rain bits and pred bits
                for tsmooth_win in self.tsmooth_wins:
                    rain_bits = self._get_rain_bits_track(i, trace, tsmooth_win)
                    self._save_rain_bits(bits_arr, tsmooth_win, rain_bits)
                    pred_bits = self._get_pred_bits_track(i, trace, tsmooth_win)
                    self._save_pred_bits(bits_arr, tsmooth_win, pred_bits)
                
                self.y_large[self.example_cnt] = bits_arr
                self.example_cnt += 1 
                example += 1
                
        print 'Done.', time() - t0, 'seconds'
        print 'example_cnt =', self.example_cnt
        print 'total, usable, example =', total, usable, example

    def test_random_examples(self):
        print 'Testing random examples ...'
        t0 = time()
        
        monitor = Monitor(self.pred_len, self.interval_len) # monitor for the current month
        example_cnt = 0
        for i in range(self.clip_shape[0], self.matrix.shape[0] - self.pred_len):
            if not self.usable(i):
                continue
            clipper = Clipper(self.matrix[i + 1 -self.clip_shape[0] : i + 1], 
                              self.flow[i + 1 -self.clip_shape[0] : i + 1], 
                              self.image_border, self.clip_shape)
            for _ in range(self.examples_per_image):
                center = self.get_random_center()
                rain_probs, trace = self.predictor.predict(clipper, center)
                example_cnt += 1
                if trace is None:
                    continue
                rain_bits = self._get_rain_bits(i, trace, self.tsmooth_win)
                pred_bits = self._get_pred_bits(i, trace, self.tsmooth_win)
                monitor.check_prediction(rain_bits, pred_bits, rain_probs)
                self.monitor.check_prediction(rain_bits, pred_bits, rain_probs)
                
        monitor.print_results()
        print 'Done.', example_cnt, 'examples,', time() - t0, 'seconds'
        
    def test_flow(self):
        tp = (self.y * self.y_pred).mean(axis=0)
        pred = self.y_pred.mean(axis=0)
        rain = self.y.mean(axis=0)
        precision = tp / pred
        recall = tp / rain
        f1 = 2 * precision * recall / (precision + recall)
        print precision, ':precision'
        print recall, ':recall'
        print f1, ':f1'
        
    def create_pred_stat(self):
        pred_stat = {}
        for start in range(0, self.pred_len, self.interval_len):
            stop = start + self.interval_len
            pred_stat[stop] = {'n_pred' : 0, 'n_rain' : 0, 'n_pred_and_rain' : 0, 
                               'n_error' : 0, 'precision' : 0., 'recall' : 0., 'f1' : 0.}
        return pred_stat
    
    def calc_pred_stat(self, pred_stat):
        for start in range(0, self.pred_len, self.interval_len):
            stop = start + self.interval_len
            d = pred_stat[stop]
            d['n_error'] = d['n_pred'] + d['n_rain'] - d['n_pred_and_rain'] * 2
            d['precision'] = d['n_pred_and_rain'] * 1. / d['n_pred']
            d['recall'] = d['n_pred_and_rain'] * 1. / d['n_rain']
            d['f1'] = d['precision'] * d['recall'] * 2. / (d['precision'] + d['recall'])
            
            print("%d:\t%.3f\t%.3f\t%.3f" % (stop, d['precision'], d['recall'], d['f1']))
#        pprint(pred_stat)
        
    def check_flow_prediction(self, i, row, col, rows, cols, flow_pred_stat):
        for start in range(0, self.pred_len, self.interval_len):
            stop = start + self.interval_len
            rain_seq = self.matrix[i + 1 + start : i + 1 + stop, row, col]
            pred_seq = self.matrix[i, rows[start : stop], cols[start : stop]]
#            center_near = (rows[start], cols[start])
#            center_far = (rows[stop-1], cols[stop-1])
#            last_frame = cv.fromarray(self.matrix[i])
#            li = cv.InitLineIterator(last_frame, (center_near[1], center_near[0]), 
#                                     (center_far[1], center_far[0]))
#            pred_seq = [p for p in li]fg
#            pred_seq = np.array(pred_seq)
            
#            if self.t_smooth_win > 1:
#                rain_seq = np.convolve(rain_seq, self.t_smooth_kernel)[
#                                    (self.t_smooth_win-1):-(self.t_smooth_win-1)]
#                pred_seq = np.convolve(pred_seq, self.t_smooth_kernel)[
#                                    (self.t_smooth_win-1):-(self.t_smooth_win-1)]
            rain = rain_seq.max() >= self.threshold
            pred = pred_seq.max() >= self.threshold
            flow_pred_stat[stop]['n_pred'] += pred
            flow_pred_stat[stop]['n_rain'] += rain
            flow_pred_stat[stop]['n_pred_and_rain'] += pred * rain
        
    def build_matrix(self, raw_data_file):
        self.data_dir = '/home/yuanxy/projects/nowcasting/data/'
        f = gzip.open(self.data_dir + raw_data_file)
        matrix = np.zeros(self.video_shape, dtype='uint8')
        for i in xrange(matrix.shape[0]):
            try:
                t,pix = cPickle.load(f)
                pix *= pix < 20
                #Gaussian_pix = cv2.GaussianBlur(pix,(5,5),sigmaX=0)               
                #matrix[i] = Gaussian_pix
                if "AZ9280" in raw_data_file:  # for chengdu
                    border = [(pix.shape[0] - self.video_shape[1]) / 2,
                              (pix.shape[1] - self.video_shape[2]) / 2]
                    pix = pix[border[0]:-border[0], border[1]:-border[1]]
#                matrix[i, self.pad_border[0]:-self.pad_border[0], 
#                       self.pad_border[1]:-self.pad_border[1]] = \
#                       pix[self.image_border[0]:-self.image_border[0], 
#                           self.image_border[1]:-self.image_border[1]]
                matrix[i] = pix
            except EOFError, e:
                traceback.print_exc()
                break;
        f.close()
        return matrix
        
    def build_flow(self, matrix, reverse=False):
        flow = np.zeros(self.video_shape + (2,), dtype='int8')
#        flow_raw = np.zeros(self.video_shape + (2,), dtype='float32')
        maxval = 12.
        minval = -12.
        for i in xrange(1, flow.shape[0]):
            frame_pair = matrix[i-1:i+1]
            if reverse:
                flow_raw = get_normalized_flow(frame_pair[::-1])
                flow_raw *= -1.
            else:
                flow_raw = get_normalized_flow(frame_pair)
            
            flow_raw[flow_raw > maxval] = maxval
            flow_raw[flow_raw < minval] = minval
            flow_raw *= 10.  # in range [-120, 120]
            flow[i] = flow_raw.round().astype('int8')
        return flow
    
    def load_matrix(self, npy_file):
        self.matrix[:, self.pad_border[0]:-self.pad_border[0], 
               self.pad_border[1]:-self.pad_border[1]] = \
                np.load(npy_file)[:, self.image_border[0] : -self.image_border[0], 
                                   self.image_border[1] : -self.image_border[1]]
#        return matrix
    
    def load_flow(self, npy_flow_file):
        self.flow[:,:,:] = np.load(npy_flow_file)[:, self.image_border[0] : -self.image_border[0], 
                                   self.image_border[1] : -self.image_border[1]]
#        return flow


# from colorfulCloudsCore.py
MAX_CLEAN_RATIO = 0.5
MIN_MOVE_PIXEL = 400

win = 100;
nebr = 20;

alpha = 0.5;
  
norm_min = 2;

denoise_threshold=0.1

def norm_trans(array):
    frame = array.astype("float");
    frame = frame*(frame<16)
    frame_max = numpy.max(frame);
    if frame_max<=9 : frame_max = 9
    #print "frame_max",frame_max
    frame = frame/frame_max;
    frame = (frame*256).astype("uint8")
    return frame
  
def getflow(x_sample):
    prev_frame = norm_trans(x_sample[-2]);
    next_frame = norm_trans(x_sample[-1]);
    flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, 0.5,3,win, 3, nebr, nebr/4, cv2.OPTFLOW_FARNEBACK_GAUSSIAN);
    
    #get avg flow norm > norm_min
    # flow_norm = numpy.linalg.norm(flow, axis=2) # for numpy version >= 1.8
    #flow_norm = numpy.apply_along_axis(numpy.linalg.norm, 2, flow) # for numpy version < 1.8
    flow_norm = np.sum(flow**2, axis=2)**(1./2) # for numpy version < 1.8
    num_moved_flows = numpy.sum(flow_norm>norm_min)
    #print "num_moved_flows = ", num_moved_flows
    #print "flows mean = ", numpy.mean(flow)
    #print "flows max = ", numpy.max(flow)

    if num_moved_flows > MIN_MOVE_PIXEL:
        flow_fliter = numpy.zeros(shape=flow.shape);
        flow_fliter[:,:,0] = flow[:,:,0] * (flow_norm > norm_min)
        flow_fliter[:,:,1] = flow[:,:,1] * (flow_norm > norm_min)
        
        flow_mean = numpy.sum(flow_fliter, axis=(0,1)) / num_moved_flows
    else:
        flow_mean = numpy.array([0,0])

    #print "avg flow = ",flow_mean;

    return flow, flow_mean
  
def get_normalized_flow(x_sample):
    prev_frame = norm_trans(x_sample[-2]);
    next_frame = norm_trans(x_sample[-1]);
    flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, 0.5,3,win, 3, nebr, nebr/4, cv2.OPTFLOW_FARNEBACK_GAUSSIAN);
    
    #get avg flow norm > norm_min
    # flow_norm = numpy.linalg.norm(flow, axis=2) # for numpy version >= 1.8
    #flow_norm = numpy.apply_along_axis(numpy.linalg.norm, 2, flow) # for numpy version < 1.8
    flow_norm = np.sum(flow**2, axis=2)**(1./2) # for numpy version < 1.8
    num_moved_flows = numpy.sum(flow_norm>norm_min)
    #print "num_moved_flows = ", num_moved_flows
    #print "flows mean = ", numpy.mean(flow)
    #print "flows max = ", numpy.max(flow)

    if num_moved_flows > MIN_MOVE_PIXEL:
        flow_fliter = numpy.zeros(shape=flow.shape);
        flow_fliter[:,:,0] = flow[:,:,0] * (flow_norm > norm_min)
        flow_fliter[:,:,1] = flow[:,:,1] * (flow_norm > norm_min)
        
        flow_mean = numpy.sum(flow_fliter, axis=(0,1)) / num_moved_flows
    else:
        flow_mean = numpy.array([0,0])

    #print "avg flow = ",flow_mean;
    flow_mean_norm = np.sum(flow_mean**2)**(1./2)
    if flow_mean_norm > MIN_FLOW_NORM:
        flow_norm = flow_norm.reshape((flow_norm.shape[0], flow_norm.shape[1], 1)) 
        flow = flow * (flow_norm < MIN_FLOW_NORM) * flow_mean_norm / flow_norm + flow * (flow_norm >= MIN_FLOW_NORM)
    return flow

# from api.py
MIN_FLOW_NORM=2