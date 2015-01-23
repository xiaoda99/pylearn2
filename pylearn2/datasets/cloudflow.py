import time
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

default_data_files = ['radar_img_matrix_AZ9280_201407_uint8.pkl.gz',
                       'radar_img_matrix_AZ9280_201408_uint8.pkl.gz',
                       'radar_img_matrix_AZ9280_201409_uint8.pkl.gz',
                       'radar_img_matrix_AZ9010_201406_uint8.pkl.gz',
                       'radar_img_matrix_AZ9010_201407_uint8.pkl.gz',
                       'radar_img_matrix_AZ9010_201408_uint8.pkl.gz',
                       'radar_img_matrix_AZ9010_201409_uint8.pkl.gz',
                       'radar_img_matrix_AZ9200_201406_uint8.pkl.gz',
                       'radar_img_matrix_AZ9200_201407_uint8.pkl.gz',
                       'radar_img_matrix_AZ9200_201408_uint8.pkl.gz',
                       'radar_img_matrix_AZ9200_201409_uint8.pkl.gz',
                       ]

class CLOUDFLOW(dense_design_matrix.DenseDesignMatrix):
    matrix = None
    flow = None
    X_large = {}
    y_large = {}
    
    def __init__(self,  
                 which_set,
                 num_examples,
                 threshold = 3,
                 prediv = 2,
                 postdiv = 2,
                 tdiv = 2,
                 train_frame_size = (3,30,30),
                 filter_frame_size = (1,30,30),
                 predict_frame_size = (1,1,1),
                 predict_interval = 2,
                 tstride = 1,
                 data_files = default_data_files,
                 axes=('c', 0, 1, 'b'),
                 examples_per_image = None,
                 video_shape = (7200, 477, 477),
                 image_border=(88, 88),
                 pad_border=(40, 40),
                 train_slot=50,   # 5 hours
                 valid_slot=20,   # 2 hours
                 test_slot=30,   # 3 hours
                 predict_style='interval',
                 track=True,
                 frame_diff=False,
                 intensity_normalization=False,
                 max_intensity = 15.,
                 train_int_range = [0., 15.],
                 test_int_range = [0., 15.],
                 sampling_rates=(1., 1., 1., 1.),
                 rain_index_threshold=1.,
                 adaptive_sampling=True,
                 sample_prob=10.,
                 run_test=False,
                 model_file='low_intensity_ceiling4_sr0.6_best.pkl'
                 ):
        if test_int_range is None:
            test_int_range = [0., 15.] 
        else:
            if which_set != 'test':
                assert test_int_range == [0., 15.]
        if train_int_range is None:
            train_int_range = [0., 15.] 
        else:
            if which_set == 'test':
                assert train_int_range == [0., 15.]
            
        assert predict_style in ['interval', 'point']
        self.__dict__.update(locals())
        del self.self
        print '\nBuilding', which_set, 'set...'
        
        self.image_border = (np.ceil(image_border[0]/prediv), 
                             np.ceil(image_border[1]/prediv))
        assert self.train_frame_size[1] % 2 == 1 and self.train_frame_size[2] % 2 == 1
        self.train_frame_radius = ((self.train_frame_size[1]-1)/2, (self.train_frame_size[2]-1)/2)         
        self.init_slots()        
        self.init_defaults()
        
        nmonth = len(data_files)
        self.logical_matrix_shape = (nmonth,
                    np.ceil(video_shape[0]*1./tdiv), 
                    np.round(video_shape[1]*1./prediv - self.image_border[0]*2), 
                    np.round(video_shape[2]*1./prediv - self.image_border[1]*2))
        physical_matrix_shape = (nmonth,
                    np.ceil(video_shape[0]*1./tdiv), 
                    np.round(video_shape[1]*1./prediv - self.image_border[0]*2 + self.pad_border[0]*2), 
                    np.round(video_shape[2]*1./prediv - self.image_border[1]*2 + self.pad_border[1]*2))
        
        flow_shape = (nmonth,
                    np.ceil(video_shape[0]*1./tdiv), 
                    np.round(video_shape[1]*1./prediv/postdiv - self.image_border[0]/2*2), # flow's border is border/2
                    np.round(video_shape[2]*1./prediv/postdiv - self.image_border[1]/2*2),
                    2)
        if CLOUDFLOW.matrix is not None:
            assert CLOUDFLOW.matrix.shape == physical_matrix_shape
            assert CLOUDFLOW.flow.shape == flow_shape
        else:
            print 'Preallocating matrix and flow ...'
            CLOUDFLOW.matrix = np.zeros(physical_matrix_shape, dtype='uint8')
            CLOUDFLOW.flow = np.zeros(flow_shape, dtype='int8')
            print 'Preallocating matrix and flow done.'
            
            data_dir = '/home/yuanxy/projects/nowcasting/data/'
            for month in range(len(data_files)):
                data_file = data_files[month]
                print '\n',data_file
    #            ramdisk_root = '/mnt/ramdisk/'
                ramdisk_root = '/home/xd/ramdisk_backup/'
                npy_file = ramdisk_root + data_file.replace('.pkl.gz', '.npy')
                npy_flow_file = ramdisk_root + data_file.replace('.pkl.gz', '_flow256.npy')
                assert os.path.isfile(npy_file)
                if os.path.isfile(npy_file):
                    print 'Cached. Loading data from ramdisk...'
                    matrix = np.load(npy_file)[:, self.image_border[0] : -self.image_border[0], 
                                               self.image_border[1] : -self.image_border[1]]
                    flow = np.load(npy_flow_file)[:, self.image_border[0]/2 : -self.image_border[0]/2, 
                                               self.image_border[1]/2 : -self.image_border[1]/2]
                    #pad_width = ((0,0), (pad_border[0], pad_border[0]), (pad_border[1], pad_border[1]))
                    #self.matrix[month] = np.lib.pad(matrix, pad_width, 'constant')  # too slow
                    CLOUDFLOW.matrix[month, :, pad_border[0]:-pad_border[0], pad_border[1]:-pad_border[1]] = matrix
                    CLOUDFLOW.flow[month] = (flow.astype('int') - 128).astype('int8')
                    print 'done.'
        
        if run_test is None:
            return   # for show_random_examples()
        
        dummy_frame = np.zeros((self.train_frame_size[1], self.train_frame_size[2]))
        ds_shape = cv2.resize(dummy_frame, (0,0), fx=1./self.postdiv, fy=1./self.postdiv).shape
        self.train_dim = self.train_frame_size[0] * ds_shape[0] * ds_shape[1]
        predict_len = self.predict_frame_size[0] if predict_style == 'point' else 1
        self.predict_dim = predict_len * \
                    (self.predict_frame_size[1]) * \
                    (self.predict_frame_size[2]) 

        if self.which_set in CLOUDFLOW.X_large:
            assert CLOUDFLOW.X_large[self.which_set].shape == (num_examples, self.train_dim)
            assert CLOUDFLOW.y_large[self.which_set].shape == (num_examples, self.predict_dim)
        else:
            print 'Preallocating X and y...'
#            CLOUDFLOW.X_large[self.which_set] = np.zeros((num_examples, self.train_dim), dtype='uint8')
#            CLOUDFLOW.y_large[self.which_set] = np.zeros((num_examples, self.predict_dim), dtype='uint8')
            CLOUDFLOW.X_large[self.which_set] = np.zeros((num_examples, self.train_dim))
            CLOUDFLOW.y_large[self.which_set] = np.zeros((num_examples, self.predict_dim))
            print 'Preallocating X and y done.' 
        
        if run_test is None:
            return
        
        self.gen_random_examples2(run_test)
        
        shape = (ds_shape[0],  #rows
                 ds_shape[1],  #cols
                 self.train_frame_size[0]   #frames, i.e. channels
                 )     
        view_converter = dense_design_matrix.DefaultViewConverter(shape, self.axes)
        super(CLOUDFLOW,self).__init__(X = CLOUDFLOW.X_large[self.which_set][:self.example_cnt], 
                                        y = CLOUDFLOW.y_large[self.which_set][:self.example_cnt], 
                                        view_converter = view_converter)
        
    def init_slots(self):
        self.train_slot /= self.tdiv
        self.valid_slot /= self.tdiv
        self.test_slot /= self.tdiv
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
        start = pos-self.train_frame_size[0]-self.predict_interval - self.predict_frame_size[0] + 1
        stop = pos
        if start % self.whole_slot >= self.usable_start and \
            stop % self.whole_slot <= self.usable_stop: 
            return True
        else:
            return False
            
    def init_defaults(self):
        self.show_right = False
        self.show_right_track = None
        self.showdiv = 2
        self.model_base = '/home/xd/projects/pylearn2/pylearn2/scripts/nowcasting/batch_exp/'
#        self.model_path = self.model_base + 'threshold_3_0.6_best.pkl'
#        self.model_path_track = self.model_base + 'threshold_3_0.6_best.pkl'
        self.model_path_track = self.model_base + self.model_file
        self.model_path = self.model_path_track
        self.cnts_total = np.zeros(4, dtype='int32')
        self.cnts_sampled = np.zeros(4, dtype='int32')
        self.min_flow_norm = 4.
        self.max_flow_norm = 6.
        
    def sampled_old(self, last_rain, rain):
        if last_rain == 0 and rain == 0:
            type = 0
        elif last_rain == 0 and rain == 1:
            type = 1
        elif last_rain == 1 and rain == 0:
            type = 2
        else:
            type = 3
        
        threshold = self.sampling_rates[type]
        ret = np.random.uniform(0., 1.) < threshold
        
        self.cnts_total[type] += 1
        self.cnts_sampled[type] += ret
        return ret
    
    def compute_rain_index(self, frame, center):
        radius_fractions = [0.0, 1.]
        center = np.array(center)
        for rf in radius_fractions:
            r = center * rf
            central_area = frame[center[0]-r[0] : center[0]+r[0]+1, center[1]-r[1] : center[1]+r[1]+1]
            nrain = (central_area >= self.threshold).sum()
            if nrain > 0:
                return nrain * 1. / central_area.size
        return 0.
    
    def compute_rain_index2(self, frame, center):
        if self.rain_index_threshold == 1.:
            rain = frame[center[0], center[1]] >= self.threshold
            if rain:
                return 1.
            if (frame >= self.threshold).sum() == 0:
                return 0.
            return self.sampling_rates[0]
        else:
            radius_fractions = [0.0, 1.]
            center = np.array(center)
            for rf in radius_fractions:
                r = center * rf
                central_area = frame[center[0]-r[0] : center[0]+r[0]+1, center[1]-r[1] : center[1]+r[1]+1]
                nrain = (central_area >= self.threshold).sum()
                if nrain > 0:
                    return nrain * 1. / central_area.size
            return 0.
            
    def is_sampled(self, train_frames, target_frames, center, rain):
        train_frame = train_frames[-1]
        target_frame = target_frames.max(axis=0)
        
        rain_index = max(self.compute_rain_index2(train_frame, center), 
                             self.compute_rain_index2(target_frame, center))
        sampled = np.random.uniform(0., 1.) < rain_index / self.rain_index_threshold
    
        last_rain = train_frame[center[0], center[1]] >= self.threshold
#        rain = target_frame[center[0], center[1]] >= self.threshold
        if last_rain == 0 and rain == 0:
            type = 0
        elif last_rain == 0 and rain == 1:
            type = 1
        elif last_rain == 1 and rain == 0:
            type = 2
        else:
            type = 3
        
        self.cnts_total[type] += 1
        self.cnts_sampled[type] += sampled
        return sampled
    
    def sampled(self, train_frames, rain, rain_prob_flow, mean_intensity):
        train_frame = train_frames[-1]
#        target_frame = target_frames.max(axis=0)
#        if max(train_frame[center[0], center[1]], target_frame[center[0], center[1]]) == 1:
        if rain or rain_prob_flow:
            return True
        if train_frame.sum() == 0:
            return False
        
#        mean_intensity = self.compute_mean_intensity(train_frames)
        if self.which_set == 'test':
            return True
        elif self.adaptive_sampling:
            return np.random.uniform(0., 1.) < mean_intensity * self.sample_prob / 3.
        else:
            return np.random.uniform(0., 1.) < self.sample_prob
        
    def compare_models(self, rain, pred0, pred1, model_pair):
        if not rain and not pred0 and not pred1:
            return
        model_pair['m0_wrong'] += (pred0 != rain)
        model_pair['m1_wrong'] += (pred1 != rain)
        model_pair['both_wrong'] += (pred0 != rain) * (pred1 != rain)
        
    def record_pred_stat(self, rain, pred0, pred1, pred_stat, mean_intensity, flow_norm):
        if not rain and not pred0 and not pred1:
            return
        if pred0 == rain and pred1 == rain:
            type = 0
        elif pred0 != rain and pred1 == rain:
            type = 1
        elif pred0 == rain and pred1 != rain:
            type = 2
        else:
            type = 3
        pred_stat.append((mean_intensity, flow_norm, type))
        
    def show_pred_stat(self, pred_stat):
        pass
    
    def compute_mean_intensity(self, frames):
        return frames.sum() * 1. / (frames > 0.).sum()
        
    def gen_random_examples2(self, test_mode=False):
        print 'Generating random examples ...'
        t0 = time.time()
        
        h_center_low = self.train_frame_radius[0]
        h_center_high = self.logical_matrix_shape[2] - self.train_frame_radius[0]
        
        w_center_low = self.train_frame_radius[1]
        w_center_high = self.logical_matrix_shape[3] - self.train_frame_radius[1]
        
        #track_frames = np.zeros(self.train_frame_size)
        
        self.example_cnt = 0
        
        if test_mode:
            pred_func, pred_func_track = self.build_pred_func()
            
            model_template = {'pred':0, 'npred':0, 'nrain':0, 'npred&rain':0}
            models = {}
            models['flow'] = model_template.copy()
#            models['nn'] = model_template.copy()
            models['tracknn'] = model_template.copy()
            groups = {}
            from copy import deepcopy
            groups[0] = deepcopy(models)   # 0~2
            groups[1] = deepcopy(models)   # 2~4
#            groups[2] = deepcopy(models)   # 4~6
#            groups[3] = deepcopy(models)   # 6~8
#            groups[4] = deepcopy(models)   # 8~10
#            groups[5] = deepcopy(models)   # 10~12
            
            model_pair = {'m0_wrong':0, 'm1_wrong':0, 'both_wrong':0}
            self.pred_stat = []
        
        self.dc = []
        self.flow_norms = []
        self.mean_intensities0 = []
        self.mean_intensities1 = []
        self.mean_intensities2 = []
        for month in range(CLOUDFLOW.matrix.shape[0]):
            print 'month =', month
            for i in range(self.train_frame_size[0] + self.predict_interval + self.predict_frame_size[0] - 1, 
                            CLOUDFLOW.matrix.shape[1]):
                if not self.usable(i):
                    continue
                for _ in range(self.examples_per_image):
                    h_center = np.random.randint(h_center_low, h_center_high)
                    w_center = np.random.randint(w_center_low, w_center_high)
                    predict_frame_center = train_frame_center = (h_center, w_center)    
                    
#                    filter_frames = self.get_frames(month, i, train_frame_center, 
#                                                (self.filter_frame_size[1]/2, self.filter_frame_size[2]/2))
#                    filter_frames = filter_frames[-self.filter_frame_size[0]:]
#                    if np.sum(filter_frames >= self.threshold) < 1:
#                        continue
                    
                    flow_frame, flow_mean, flow_center = self.get_flow_frame(month, i, train_frame_center, self.train_frame_radius)
                    
                    dt = -(self.train_frame_size[0] + self.predict_interval + self.predict_frame_size[0] - 1)
                    track_frame_center = self.translate_coords(train_frame_center, flow_mean, dt)
                    diag_radius = int(math.ceil(math.sqrt(pow(self.train_frame_size[1]/2, 2) + 
                                                        pow(self.train_frame_size[2]/2, 2))))
                    if not self.in_logical_matrix(track_frame_center, (diag_radius, diag_radius)):
                        continue
                                        
                    track_frames_ext, _ = self.get_track_frames_ext(month, i, train_frame_center, flow_mean)
                    if track_frames_ext is None:
                        continue
                    track_frames = track_frames_ext[:self.train_frame_size[0]]
                    target_track_frames = track_frames_ext[-self.predict_frame_size[0]:]
#                    int_mean = self.compute_mean_intensity(track_frames)
                    
                    train_frames_ext = self.get_frames_ext(month, i, train_frame_center, self.train_frame_radius)
                    train_frames = train_frames_ext[:self.train_frame_size[0]]
                    target_frames = train_frames_ext[-self.predict_frame_size[0]:]

                    rain = target_frames[:, self.train_frame_radius[0], self.train_frame_radius[1]].max() >= self.threshold
                    rain_prob_flow, traceback_vals = self.pred_func_flow(month, i, train_frame_center, flow_mean)
                    mean_intensity = self.compute_mean_intensity(train_frames)
                    if not self.sampled(track_frames, rain, rain_prob_flow, mean_intensity):
                        continue
#                    if self.which_set == 'test' and (int_mean < self.test_int_range[0] or 
#                                                     int_mean > self.test_int_range[1]):
#                        continue
                    
#                    if not self.sampled(last_rain, rain):
#                        continue
#                    if self.intensity_normalization:
#                        train_frames = self.normalize(train_frames)
#                        track_frames = self.normalize(track_frames)
                    
                    if not test_mode:
#                        if mean_intensity < self.mean_intensity_range[0] or \
#                            mean_intensity > self.mean_intensity_range[1]:
#                            continue
                        assert self.track
                        frames = track_frames if self.track else train_frames
                        ds = cv2.resize(frames.transpose((1,2,0)), 
                                        (0,0), fx=1./self.postdiv, fy=1./self.postdiv).transpose((2,0,1))
#                        if self.frame_diff:
#                            ds = self.diff(ds)
#                        x = ds.round().astype('uint8').flatten()
                        x = ds.flatten()
                        
#                        train_int_mean = self.compute_mean_intensity(x)
#                        if train_int_mean < self.train_int_range[0] or train_int_mean > self.train_int_range[1]:
#                            continue
                        x = x * (x <= self.max_intensity) + \
                                self.max_intensity * (x > self.max_intensity)
                    
                        CLOUDFLOW.X_large[self.which_set][self.example_cnt] = x
                        CLOUDFLOW.y_large[self.which_set][self.example_cnt, 0] = rain
                        self.example_cnt += 1
                    else:
                        assert self.which_set == 'test'
                        flow_norm = np.sum(flow_mean**2)**(1./2)
                        group_id = self.group(mean_intensity, flow_norm)
                        models_in_group = groups[group_id]
                        
                        rain_prob_track = self.predict_rain_prob(track_frames, pred_func_track)
                        pred_track = rain_prob_track >= 0.5
                        models['tracknn']['pred'] = pred_track
                        models_in_group['tracknn']['pred'] = pred_track
#                        rain_prob = self.predict_rain_prob(train_frames, pred_func)
#                        models['nn']['pred'] = rain_prob >= 0.5
#                        models_in_group['nn']['pred'] = rain_prob >= 0.5
                        
#                        ensemble['pred'] = (rain_prob_track + rain_prob) / 2. >= 0.5
                        
                        rain_prob_flow, traceback_vals = self.pred_func_flow(month, i, train_frame_center, flow_mean)
                        pred_flow = rain_prob_flow >= 0.5
                        models['flow']['pred'] = pred_flow
                        models_in_group['flow']['pred'] = pred_flow
                                                
                        for model_name in models:
                            model = models[model_name]
                            model['npred'] += model['pred']
                            model['nrain'] += rain
                            model['npred&rain'] += (model['pred'] * rain)
                        
                        for model_name in models_in_group:
                            model = models_in_group[model_name]
                            model['npred'] += model['pred']
                            model['nrain'] += rain
                            model['npred&rain'] += (model['pred'] * rain)
                            
                        self.compare_models(rain, pred_flow, pred_track, model_pair)
                        mean_intensity = self.compute_mean_intensity(track_frames_ext)
                        self.record_pred_stat(rain, pred_flow, pred_track, self.pred_stat, mean_intensity, flow_norm)
                                  
            print 'example_cnt =', self.example_cnt
            print 'cnts_total =', self.cnts_total
            print 'cnts_sampled =', self.cnts_sampled   
            if test_mode:
                pprint(models)
                pprint(groups)   
              
        t1 = time.time()
        print 'Done. Total =', self.example_cnt, 'Time:', t1 - t0
        print 'cnts_total =', self.cnts_total
        print 'cnts_sampled =', self.cnts_sampled
#        print 'cnts_total_track =', self.cnts_total_track
#        print 'cnts_sampled_track =', self.cnts_sampled_track
        
        if test_mode:       
            for model_name in models:
                model = models[model_name]
                precision = model['npred&rain']*1./model['npred']
                recall = model['npred&rain']*1./model['nrain']
                model['false_pos'] = 1. - precision
                model['false_neg'] = 1. - recall
                model['f1'] = 2. * precision * recall / (precision + recall)
            for group_id in groups:
                models_in_group = groups[group_id]    
                for model_name in models_in_group:
                    model = models_in_group[model_name]
                    precision = model['npred&rain']*1./model['npred']
                    recall = model['npred&rain']*1./model['nrain']
                    model['false_pos'] = 1. - precision
                    model['false_neg'] = 1. - recall
                    model['f1'] = 2. * precision * recall / (precision + recall)
                    
#            print 'nn_wrong =', nn_wrong, 'tracknn_wrong =', tracknn_wrong, 'both_wrong =', both_wrong,
            pprint(models)
            pprint(groups) 
            print model_pair
               
    def show_random_examples(self):
        h_center_low = self.train_frame_size[1]/2*self.showdiv
        h_center_high = self.logical_matrix_shape[2] - self.train_frame_size[1]/2*self.showdiv
        
        w_center_low = self.train_frame_size[2]/2*self.showdiv
        w_center_high = self.logical_matrix_shape[3] - self.train_frame_size[2]/2*self.showdiv
        
        pred_func, pred_func_track = self.build_pred_func()
        
        nexamples = 0
        
        nn = {'name':'nn', 'pred':0, 'npred':0, 'nrain':0, 'npred&rain':0}
        tracknn = {'name':'tracknn', 'pred':0, 'npred':0, 'nrain':0, 'npred&rain':0}
        flow = {'name':'flow', 'pred':0, 'npred':0, 'nrain':0, 'npred&rain':0}
        persistent = {'name':'persistent', 'pred':0, 'npred':0, 'nrain':0, 'npred&rain':0}
        models = [nn, tracknn, flow, persistent]
        
        while True:
            month = np.random.randint(CLOUDFLOW.matrix.shape[0])
            i = np.random.randint(self.train_frame_size[0] + self.predict_interval + self.predict_frame_size[0] - 1, 
                                  CLOUDFLOW.matrix.shape[1])
            if not self.usable(i):
                continue
            
            h_center = np.random.randint(h_center_low, h_center_high)
            w_center = np.random.randint(w_center_low, w_center_high)
            predict_frame_center = train_frame_center = (h_center, w_center)
            
            flow_frame, flow_mean, flow_center = self.get_flow_frame(month, i, train_frame_center, self.train_frame_radius)
            
            dt = -(self.train_frame_size[0] + self.predict_interval + self.predict_frame_size[0] - 1)
            track_frame_center = self.translate_coords(train_frame_center, flow_mean, dt)
            diag_radius = int(math.ceil(math.sqrt(pow(self.train_frame_size[1]/2, 2) + 
                                                pow(self.train_frame_size[2]/2, 2))))
            if not self.in_logical_matrix(track_frame_center, (diag_radius, diag_radius)):
                continue
        
            track_frames_ext, track_frames_ext_show = self.get_track_frames_ext(month, i, train_frame_center, flow_mean)
            if track_frames_ext is None:
                continue
            track_frames = track_frames_ext[:self.train_frame_size[0]]
            
            train_frames_ext = self.get_frames_ext(month, i, train_frame_center, self.train_frame_radius)
            train_frames = train_frames_ext[:self.train_frame_size[0]]
            target_frames = train_frames_ext[-self.predict_frame_size[0]:]

            rain = target_frames[:, self.train_frame_radius[0], self.train_frame_radius[1]].max() >= self.threshold
            if not self.is_sampled(track_frames, target_frames, self.train_frame_radius, rain):
                continue
            
            assert self.which_set == 'test'
            rain_prob_track = self.predict_rain_prob(track_frames, pred_func_track)
            pred_track = rain_prob_track >= 0.5
#            rain_prob = self.predict_rain_prob(train_frames, pred_func)
#            nn['pred'] = rain_prob >= 0.5
            rain_prob_flow, traceback_vals = self.pred_func_flow(month, i, train_frame_center, flow_mean)   
            pred_flow = rain_prob_flow >= 0.5 
            
            ds = cv2.resize(track_frames_ext.transpose((1,2,0)), 
                            (0,0), fx=1./self.postdiv, fy=1./self.postdiv).transpose((2,0,1))
#            track_frames_ext_ds = ds.round().astype('uint8')
            track_frames_ext_ds = ds
                
#            c0 = True if self.show_right is None else (nn['pred'] == rain) == self.show_right
#            c1 = True if self.show_right_track is None else (tracknn['pred'] == rain) == self.show_right_track
#            if c0 and c1:
#            if tracknn['pred'] != rain:
#            if not self.filtered(last_rain, rain, rain_prob_track, flow_mean, month, i):
#                continue
            if not rain and not pred_track and not pred_flow:
                continue
            mean_intensity0 = self.compute_mean_intensity(train_frames_ext)
            mean_intensity = self.compute_mean_intensity(track_frames)
            mean_intensity_ds = self.compute_mean_intensity(track_frames_ext_ds[:self.train_frame_size[0]])
            if mean_intensity > 3.:
                continue
            
            print '\n'
            print 'rain =', rain, 'tracknn[prob] =', rain_prob_track, 'flow[prob] =', pred_flow
            print 'flow_mean =', flow_mean
            print 'mean_intensities :', mean_intensity0, mean_intensity, mean_intensity_ds
            print 'center_vals =', train_frames_ext[-self.predict_frame_size[0]:, 
                                                    self.train_frame_radius[0], 
                                                    self.train_frame_radius[1]]
            print 'traceback vals =', traceback_vals
            
            ds_radius = (self.train_frame_radius[0] * self.showdiv, self.train_frame_radius[1] * self.showdiv)
            train_frames_ext_ds = self.get_frames_ext(month, i, train_frame_center, ds_radius)
            train_frames_ext_ds = cv2.resize(train_frames_ext_ds.transpose((1,2,0)), (0,0), 
                                    fx=1./self.showdiv, fy=1./self.showdiv).transpose((2,0,1))
            pv = patch_viewer.PatchViewer(grid_shape=(4, train_frames_ext.shape[0]), 
                                          patch_shape=[train_frames_ext.shape[1], train_frames_ext.shape[2]])
            for fidx in range(train_frames_ext.shape[0]):
                pv.add_patch(self.rescale(train_frames_ext[fidx]), activation=0, rescale=False)
                
            for fidx in range(train_frames_ext_ds.shape[0]):
                pv.add_patch(self.rescale(train_frames_ext_ds[fidx]), activation=0, rescale=False)             
                
            for fidx in range(track_frames_ext_show.shape[0]):
                pv.add_patch(self.rescale(track_frames_ext_show[fidx]), activation=0, rescale=False)
                  
            for fidx in range(track_frames_ext_show.shape[0]):
                pv.add_patch(self.rescale(track_frames_ext_ds[fidx]), activation=0, rescale=False)
#                    
#                for fidx in range(track_frames_ds.shape[0]):
#                    pv.add_patch(track_frames_ds[fidx], activation=0)
#                for fidx in range(future_track_frames_ds.shape[0]):
#                    pv.add_patch(future_track_frames_ds[fidx], activation=0)
                
            pv.show()
      
    def normalize(self, frames):
        max = frames.max()
        frames = frames.astype('float') / max
        frames *= 10.
        return frames
    
    def diff(self, frames):
        for i in range(frames.shape[0] - 1):
            frames[i] = frames[i + 1] - frames[i]
        return frames
                  
    def filtered(self, last_rain, rain, rain_prob_track, flow_mean, month, i):
        flow_norm = np.sum(flow_mean**2)**(1./2)
        ret = (not last_rain) and (rain or rain_prob_track >= 0.5) and \
            flow_norm >= self.min_flow_norm and flow_norm <= self.max_flow_norm
        if ret:
            print 'month =', month, 'i =', i
            print 'flow_mean =', flow_mean, 'flow_mean_norm =', flow_norm
        return ret
    
    def group(self, mean_intensity, flow_norm):
        if mean_intensity <= 3.:
            return 0
        else:
            return 1
#        flow_norm = np.sum(flow_mean**2)**(1./2)
        group_id = int(flow_norm / 2.)
        if group_id > 5:
            group_id = 5
        return group_id
                                  
    def rescale(self, frame):
        max_intensity = 10.
        frame /= max_intensity
        frame = frame * (frame <= 1.) + 1. * (frame > 1.)
        frame = frame * (frame >= -1) - 1. * (frame < -1.)
        return frame
    
    def in_logical_matrix(self, center, radius):
        if center[0] - radius[0] >= 0 and center[0] + radius[0] < self.logical_matrix_shape[2] and \
            center[1] - radius[1] >= 0 and center[1] + radius[1] < self.logical_matrix_shape[3]:
            return True
        else:
            return False
            
    def predict_rain_prob(self, frames, pred_func):    
        ds = cv2.resize(frames.transpose((1,2,0)), 
                        (0,0), fx=1./self.postdiv, fy=1./self.postdiv).transpose((2,0,1))
#        x = ds.round().astype('uint8').flatten()
        x = ds.flatten()
        rain_prob = pred_func(x.reshape(1, x.shape[0]))[0][0]
        return rain_prob
                                     
    def get_rain_status(self, month, i, center, last_rain):  
        assert self.predict_style == 'interval'
        if self.predict_style == 'interval':
#            if last_rain == 0:
            if True:
                rain = CLOUDFLOW.matrix[month, 
                               i-self.predict_frame_size[0]+1:i+1, 
                               self.pad_border[0]+center[0],
                               self.pad_border[1]+center[1]
                            ].max() >= self.threshold
            else:
                rain = CLOUDFLOW.matrix[month, 
                               i-self.predict_frame_size[0]+1:i+1, 
                               self.pad_border[0]+center[0],
                               self.pad_border[1]+center[1]
                            ].min() >= self.threshold
        else:
            for j in range(self.predict_frame_size[0]):
                rain = CLOUDFLOW.matrix[month,
                               i - self.predict_frame_size[0] + 1 + j, 
                               self.pad_border[0]+center[0], 
                               self.pad_border[1]+center[1]] >= self.threshold
        return rain
                               
    def get_frames(self, month, i, center, radius):
        if self.pad_border[0] + center[0] - radius[0] >= 0 and \
                self.pad_border[0] + center[0] + radius[0] < CLOUDFLOW.matrix.shape[2] and \
                self.pad_border[1] + center[1] - radius[1] >= 0 and \
                self.pad_border[1] + center[1] + radius[1] < CLOUDFLOW.matrix.shape[3]:
            return CLOUDFLOW.matrix[month,
                        i-self.train_frame_size[0]-self.predict_interval-self.predict_frame_size[0]+1:
                        i-self.predict_interval-self.predict_frame_size[0]+1,
                        self.pad_border[0]+center[0]-radius[0] : self.pad_border[0]+center[0]+radius[0]+1,
                        self.pad_border[1]+center[1]-radius[1] : self.pad_border[1]+center[1]+radius[1]+1
                    ].astype('float')
        else:
            return None
        
    def get_flow_frame(self, month, i, center, radius):
        assert radius[0] % self.postdiv == 0 and radius[1] % self.postdiv == 0
        flow_frame = CLOUDFLOW.flow[month,
                    i-self.predict_interval-self.predict_frame_size[0],
                    center[0]/self.postdiv - radius[0]/self.postdiv:
                    center[0]/self.postdiv + radius[0]/self.postdiv+1,
                    center[1]/self.postdiv - radius[1]/self.postdiv:
                    center[1]/self.postdiv + radius[1]/self.postdiv+1
                ].astype('float') / 10.
        flow_mean = flow_frame.mean(axis=(0,1))
#        flow_mean_norm = np.sum(flow_mean**2)**(1./2)
        flow_center = flow_frame[radius[0]/self.postdiv, 
                                 radius[1]/self.postdiv]
        return flow_frame, flow_mean, flow_center
        
    # used by show_random_examples()
    def get_frames_ext(self, month, i, center, radius):
        if self.pad_border[0] + center[0] - radius[0] >= 0 and \
                self.pad_border[0] + center[0] + radius[0] < CLOUDFLOW.matrix.shape[2] and \
                self.pad_border[1] + center[1] - radius[1] >= 0 and \
                self.pad_border[1] + center[1] + radius[1] < CLOUDFLOW.matrix.shape[3]:
            return CLOUDFLOW.matrix[month,
                        i-self.train_frame_size[0]-self.predict_interval-self.predict_frame_size[0]+1:
                        i+1,
                        self.pad_border[0]+center[0]-radius[0] : self.pad_border[0]+center[0]+radius[0]+1,
                        self.pad_border[1]+center[1]-radius[1] : self.pad_border[1]+center[1]+radius[1]+1
                    ].astype('float')
        else:
            return None
     
    def get_point_value(self, month, i, point_coords):
        return CLOUDFLOW.matrix[month, i, self.pad_border[0]+point_coords[0], self.pad_border[1]+point_coords[1]]
        
    def translate_coords(self, point_coords, flow, dt):          
        dx = flow[1] * dt * self.tdiv / self.prediv
        dy = flow[0] * dt * self.tdiv / self.prediv
        return (point_coords[0] + int(round(dx)), point_coords[1] + int(round(dy)))
    
    def pred_func_flow(self, month, i, center, flow):
        dt_near = -(self.predict_interval + 1)
        dt_far = -(self.predict_interval + self.predict_frame_size[0])
        center_near = self.translate_coords(center, flow, dt_near)
        center_far = self.translate_coords(center, flow, dt_far)
        last_frame = CLOUDFLOW.matrix[month,
                        i-self.predict_interval-self.predict_frame_size[0],
                        self.pad_border[0] : -self.pad_border[0],
                        self.pad_border[1] : -self.pad_border[1]
                    ].astype('float')
        last_frame = cv.fromarray(last_frame)
        li = cv.InitLineIterator(last_frame, (center_near[1], center_near[0]), (center_far[1], center_far[0]))
        pred = 0
        traceback_vals = []
        for p in li:
            traceback_vals.append(p)
            if p >= self.threshold:
                pred = 1
        assert traceback_vals != []
        return pred, traceback_vals
        
    def pred_func_flow_old(self, train_frames, flow):
        dt_near = -(self.predict_interval + 1)
        dt_far = -(self.predict_interval + self.predict_frame_size[0])
        center_near = self.translate_coords(self.train_frame_radius, flow, dt_near)
        center_far = self.translate_coords(self.train_frame_radius, flow, dt_far)
        print 'flow =', flow, center_near, '->', center_far
        last_frame = cv.fromarray(train_frames[-1])
        li = cv.InitLineIterator(last_frame, (center_near[1], center_near[0]), (center_far[1], center_far[0]))
        pred = 0
        traceback_vals = []
        for p in li:
            traceback_vals.append(p)
            if p >= self.threshold:
                pred = 1
        return pred, traceback_vals
    
    def get_track_frames_norotate(self, month, i, center, flow_mean):
        track_frames = np.zeros(self.train_frame_size)
        for j in range(track_frames.shape[0]):
            dt = -(self.train_frame_size[0] + self.predict_interval + self.predict_frame_size[0] - 1 - j)
            track_frame_center = self.translate_coords(center, flow_mean, dt)
            track_frames[j] = self.get_frames(month, i, track_frame_center, self.train_frame_radius)[j]
        return track_frames
    
    def get_base_radius(self, flow_mean):
        dt = self.train_frame_size[0] + self.predict_interval + self.predict_frame_size[0] - 1
        dx = flow_mean[1] * dt * self.tdiv / self.prediv
        dy = flow_mean[0] * dt * self.tdiv / self.prediv
        dx = int(math.ceil(abs(dx)))
        dy = int(math.ceil(abs(dy)))
        diag_radius = int(math.ceil(math.sqrt(pow(self.train_frame_size[1]/2, 2) + 
                                            pow(self.train_frame_size[2]/2, 2))))
        rmax = max(dx + diag_radius, dy + diag_radius)
        rmax = int(math.ceil(rmax * math.sqrt(2.)))
        radius = (rmax, rmax)
        return radius
             
    def get_track_frames(self, month, i, center, flow_mean):
        radius = self.get_base_radius(flow_mean)
        base_frames = self.get_frames(month, i, center, radius)
        if base_frames is None:
            return None
        
        flow_mean = flow_mean.reshape((1,2))
        mag, ang = cv2.cartToPolar(flow_mean[:,0], flow_mean[:,1], angleInDegrees=True)
        angle = ang[0,0]
        rot_mat = cv2.getRotationMatrix2D((radius[1], radius[0]), angle, 1.0)
        center0 = radius
        flow = (mag[0,0], 0.)
        
        track_frames = np.zeros(self.train_frame_size)
        for j in range(track_frames.shape[0]):
            frame = base_frames[j]
            rotated = cv2.warpAffine(frame, rot_mat, (frame.shape[1], frame.shape[0]))
            dt_near = -(self.train_frame_size[0] + self.predict_interval - j)
            dt_far = -(self.train_frame_size[0] + self.predict_interval + self.predict_frame_size[0] - 1 - j)
            center_near = self.translate_coords(center0, flow, dt_near)
            center_far = self.translate_coords(center0, flow, dt_far)
            r = self.train_frame_radius
            cropped = rotated[center_near[0] - r[0] : center_near[0] + r[0] + 1, 
                            center_far[1] - r[1] : center_near[1] + r[1] + 1]
            track_frames[j] = cv2.resize(cropped, (cropped.shape[0], cropped.shape[0]))
        return track_frames
            
    def get_track_frames_old(self, month, i, center, flow_mean):
        radius = self.get_base_radius(flow_mean)
        base_frames = self.get_frames(month, i, center, radius)
        if base_frames is None:
            return None
        
        flow_mean = flow_mean.reshape((1,2))
        mag, ang = cv2.cartToPolar(flow_mean[:,0], flow_mean[:,1], angleInDegrees=True)
        angle = ang[0,0]
        rot_mat = cv2.getRotationMatrix2D((radius[1], radius[0]), angle, 1.0)
        center0 = radius
        flow = (mag[0,0], 0.)
        
        track_frames = np.zeros(self.train_frame_size)
        for j in range(track_frames.shape[0]):
            frame = base_frames[j]
            rotated = cv2.warpAffine(frame, rot_mat, (frame.shape[1], frame.shape[0]))
            dt_near = -(self.train_frame_size[0] + self.predict_interval - j)
            dt_far = -(self.train_frame_size[0] + self.predict_interval + self.predict_frame_size[0] - 1 - j)
            center_near = self.translate_coords(center0, flow, dt_near)
            center_far = self.translate_coords(center0, flow, dt_far)
            r = self.train_frame_radius
            cropped = rotated[center_near[0] - r[0] : center_near[0] + r[0], 
                            center_far[1] - r[1] : center_near[1] + r[1]]
            track_frames[j] = cv2.resize(cropped, (cropped.shape[0], cropped.shape[0]))
        return track_frames
    
    def get_track_frames_ext(self, month, i, center, flow_mean):
        radius = self.get_base_radius(flow_mean)
        base_frames = self.get_frames_ext(month, i, center, radius)
        if base_frames is None:
            return None, None
#        if base_frames.sum() > 0:
#            self.mean_intensities0.append(self.compute_mean_intensity(base_frames))################
        
        flow_mean = flow_mean.reshape((1,2))
        mag, ang = cv2.cartToPolar(flow_mean[:,0], flow_mean[:,1], angleInDegrees=True)
        angle = ang[0,0]
        rot_mat = cv2.getRotationMatrix2D((radius[1], radius[0]), angle, 1.0)
        center0 = radius
        flow = (mag[0,0], 0.)
        
        track_frames = np.zeros((self.train_frame_size[0] + self.predict_interval + self.predict_frame_size[0], 
                                 self.train_frame_size[1],
                                 self.train_frame_size[2]))
        track_frames_show = np.zeros((self.train_frame_size[0] + self.predict_interval + self.predict_frame_size[0], 
                                 self.train_frame_size[1],
                                 self.train_frame_size[2]))
        for j in range(track_frames.shape[0]):
            frame = base_frames[j]
            rotated = cv2.warpAffine(frame, rot_mat, (frame.shape[1], frame.shape[0]))
#            if rotated.sum() > 0:
#                self.mean_intensities1.append(self.compute_mean_intensity(rotated))###############
            dt_near = -(self.train_frame_size[0] + self.predict_interval - j)
            dt_far = -(self.train_frame_size[0] + self.predict_interval + self.predict_frame_size[0] - 1 - j)
            center_near = self.translate_coords(center0, flow, dt_near)
            center_far = self.translate_coords(center0, flow, dt_far)
            r = self.train_frame_radius
            cropped = rotated[center_near[0] - r[0] : center_near[0] + r[0] + 1, 
                            center_far[1] - r[1] : center_near[1] + r[1] + 1]
            track_frames[j] = cv2.resize(cropped, (cropped.shape[0], cropped.shape[0]))
#            if track_frames[j].sum() > 0:
#                self.mean_intensities2.append(self.compute_mean_intensity(track_frames[j]))###################
            
            # mark two center on track_frames_show
            rotated[center_near[0], center_near[1]] = -100.
            rotated[center_far[0], center_far[1]] = -100.
            cropped_show = rotated[center_near[0] - r[0] : center_near[0] + r[0] + 1, 
                            center_far[1] - r[1] : center_near[1] + r[1] + 1]
            track_frames_show[j] = cv2.resize(cropped_show, (cropped.shape[0], cropped.shape[0]))
        return track_frames, track_frames_show
                   
    def build_pred_func(self):
        model = serial.load(self.model_path)
        print 'compiling function ...'
        X = model.get_input_space().make_batch_theano()
        y = model.fprop(X)
        func = function([X], y,allow_input_downcast=True)
        print 'done.'
        
        model = serial.load(self.model_path_track)
        print 'compiling function ...'
        X = model.get_input_space().make_batch_theano()
        y = model.fprop(X)
        func_track = function([X], y,allow_input_downcast=True)
        print 'done.'
        return func, func_track
