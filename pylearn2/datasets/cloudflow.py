import time
import os, cPickle, logging
import traceback
_logger = logging.getLogger(__name__)

import numpy
np = numpy
import gzip
import cv2
import random

from theano import function
from pylearn2.datasets import dense_design_matrix
from pylearn2.gui import patch_viewer
from pylearn2.utils import serial

class CLOUDFLOW2(dense_design_matrix.DenseDesignMatrix):
    def __init__(self,  
                 which_set,
                 num_examples,
                 threshold = 3,
                 pixnum_threshold = 1,
                 prediv = 2,
                 postdiv = 2,
                 tdiv = 2,
                 train_frame_size = (3,30,30),
                 filter_frame_size = (1,30,30),
                 predict_frame_size = (1,1,1),
                 predict_interval = 2,
                 stride = (3,3),
                 tstride = 1,
                 data_files = [],
                 axes=('c', 0, 1, 'b'),
                 examples_per_image = None,
                 video_shape = (7200, 477, 477),
                 image_border=(88, 88),
                 train_slot=50,   # 5 hours
                 valid_slot=20,   # 2 hours
                 test_slot=30,   # 3 hours
                 predict_style='point',
                 track=True,
                 ):

        assert predict_style in ['interval', 'point']
        self.__dict__.update(locals())
        del self.self
        print '\nBuilding', which_set, 'set...'
        
        self.image_border = (np.ceil(image_border[0]/prediv), 
                             np.ceil(image_border[1]/prediv))

        self.init_slots()
        
        print 'Preallocating matrix and flow ...'
        nmonth = len(data_files)
        matrix_shape = (nmonth,
                    np.ceil(video_shape[0]*1./tdiv), 
                    np.round(video_shape[1]*1./prediv - self.image_border[0]*2), 
                    np.round(video_shape[2]*1./prediv - self.image_border[1]*2))
        self.matrix = np.zeros(matrix_shape, dtype='uint8')
        
        flow_shape = (nmonth,
                    np.ceil(video_shape[0]*1./tdiv), 
                    np.round(video_shape[1]*1./prediv/postdiv - self.image_border[0]/2*2), # flow's border is border/2
                    np.round(video_shape[2]*1./prediv/postdiv - self.image_border[1]/2*2),
                    2)
        self.flow = np.zeros(flow_shape, dtype='int8')
        print 'Preallocating matrix and flow done.'
        
        self.train_dim = self.train_frame_size[0]*\
                    (self.train_frame_size[1])/self.postdiv*\
                    (self.train_frame_size[2])/self.postdiv
        predict_len = self.predict_frame_size[0] if predict_style == 'point' else 1
        self.predict_dim = predict_len * \
                    (self.predict_frame_size[1]) * \
                    (self.predict_frame_size[2]) 

        print 'Preallocating X and y...'
        X = np.zeros((num_examples, self.train_dim), dtype='uint8')
        y = np.zeros((num_examples, self.predict_dim), dtype='uint8')
        print 'Preallocating X and y done.' 

        shape = (self.train_frame_size[1] / self.postdiv,  #rows
                 self.train_frame_size[2] / self.postdiv,  #cols
                 self.train_frame_size[0]   #frames, i.e. channels
                 )     
        view_converter = dense_design_matrix.DefaultViewConverter(shape, self.axes)
        super(CLOUDFLOW2,self).__init__(X = X, y = y, view_converter = view_converter)
        
        data_dir = '/home/yuanxy/projects/nowcasting/data/'
        for month in range(len(data_files)):
            data_file = data_files[month]
            print '\n',data_file
            ramdisk_root = '/mnt/ramdisk/'
            npy_file = ramdisk_root + data_file.replace('.pkl.gz', '.npy')
            npy_flow_file = ramdisk_root + data_file.replace('.pkl.gz', '_flow256.npy')
            if os.path.isfile(npy_file):
                print 'Cached. Loading data from ramdisk...'
                matrix = np.load(npy_file)[:, self.image_border[0] : -self.image_border[0], 
                                           self.image_border[1] : -self.image_border[1]]
                self.matrix[month] = matrix
                
                flow = np.load(npy_flow_file)[:, self.image_border[0]/2 : -self.image_border[0]/2, 
                                           self.image_border[1]/2 : -self.image_border[1]/2]
                self.flow[month] = (flow.astype('int') - 128).astype('int8')
                print 'done.'
            else:
                print 'Loading data from disk and computing flow...'
                t0 = time.time()
                f = gzip.open(data_dir + data_file)
                #matrix = np.zeros(video_shape, dtype='uint8')
                matrix *= 0
                flow *= 0.0
                for i in xrange(video_shape[0]):
                    try:
                        t,pix = cPickle.load(f)
                        pix = pix * (pix < 255)
                        #Gaussian_pix = cv2.GaussianBlur(pix,(5,5),sigmaX=0)               
                        #matrix[i] = Gaussian_pix
                        if "AZ9280" in data_file:  # for chengdu
                            border = [(pix.shape[0] - video_shape[1]) / 2,
                                      (pix.shape[1] - video_shape[2]) / 2]
                            pix = pix[border[0]:pix.shape[0]-border[0],
                                      border[1]:pix.shape[1]-border[1]]
                        matrix[i] = pix
                        
                        if i == 0:
                            continue
#                        flow = flow[image_border[0]:video_shape[1]-image_border[0],
#                                    image_border[1]:video_shape[2]-image_border[1],
#                                    :]
                        flow_i, _ = get_normalized_flow(matrix[i-1:i+1])
                                
                        # downsample inplace to save memory
                        flow[i] = cv2.resize(flow_i, (0, 0), fx=1./prediv/postdiv, fy=1./prediv/postdiv) 
                    except Exception, e:
                        traceback.print_exc()
                        break;
                f.close()
                self.matrix = matrix
                t1 = time.time()
                print 'done.', t1 - t0, 'seconds'     
            
                if prediv != 1:
                    print 'Downsampling...'
                    ds *= 0.0
                    for i in range(video_shape[0]):
                        ds[i] = cv2.resize(matrix[i].astype('float'), (0, 0), fx=1./prediv, fy=1./prediv)
                    self.matrix = ds.reshape((ds_shape[0], tdiv, ds_shape[1], ds_shape[2])
                                    ).mean(axis=1).round().astype('uint8')

#                    flowds = flow # spatial downsampling has already been done
#                    self.flow = (flowds.reshape((flowds.shape[0]/tdiv, tdiv, flowds.shape[1], flowds.shape[2], flowds.shape[3])).mean(axis=1)
#                                 + 8.).clip(min=0.0).round().astype('uint8') 
                    flowds = flow.reshape((flow.shape[0]/tdiv, tdiv, flow.shape[1], flow.shape[2], flow.shape[3])).mean(axis=1)
                    maxval = 12.
                    minval = -12.
                    flowds[flowds > maxval] = maxval
                    flowds[flowds < minval] = minval
                    flowds = (flowds * 10.) + 128.  # in range [128-120, 128+120]
                    self.flow = flowds.round().astype('uint8')
                    print 'done.'
                
                print 'Caching data to ramdisk...'
                np.save(npy_file, self.matrix)
                np.save(npy_flow_file, self.flow)
                print 'done.'
                
        self.gen_random_examples()
        
        assert not np.any(np.isnan(self.X))
        self.init_defaults()
        
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
        self.model_base = '/home/xd/projects/pylearn2/pylearn2/scripts/nowcasting/'
        self.model_path = self.model_base + 'binary_skip2_3x24x24-200-1x1_best.pkl'
        self.model_path_track = self.model_base + 'tracking_36m_3x24x24-200-1x1_best.pkl'
        
    def gen_random_examples(self):
        print 'Generating random examples ...'
        t0 = time.time()
        
        h_center_low = self.train_frame_size[1]/2
        h_center_high = self.matrix.shape[2] - self.train_frame_size[1]/2
        
        w_center_low = self.train_frame_size[2]/2
        w_center_high = self.matrix.shape[3] - self.train_frame_size[2]/2
        
        track_frames = np.zeros(self.train_frame_size)
        
        self.example_cnt = 0
        for month in range(self.matrix.shape[0]):
            print 'month', month
            for i in range(self.train_frame_size[0] + self.predict_interval + self.predict_frame_size[0] - 1, 
                            self.matrix.shape[1]):
                if not self.usable(i):
                    continue
                
                for _ in range(self.examples_per_image):
                    h_center = np.random.randint(h_center_low, h_center_high)
                    w_center = np.random.randint(w_center_low, w_center_high)
                    predict_frame_center = train_frame_center = (h_center, w_center)    
                    
                    filter_frames = self.matrix[month,
                                i-self.filter_frame_size[0]-self.predict_interval-self.predict_frame_size[0]+1:
                                i-self.predict_interval-self.predict_frame_size[0]+1,
                                train_frame_center[0]-self.filter_frame_size[1]/2:
                                train_frame_center[0]+self.filter_frame_size[1]/2,
                                train_frame_center[1]-self.filter_frame_size[2]/2:
                                train_frame_center[1]+self.filter_frame_size[2]/2
                            ].astype('float') 
                    if np.sum(filter_frames >= self.threshold) < self.pixnum_threshold:
                        continue
        
                    flow_frame = self.flow[month,
                            i-self.predict_interval-self.predict_frame_size[0],
                            train_frame_center[0]/self.postdiv - self.train_frame_size[1]/2/self.postdiv:
                            train_frame_center[0]/self.postdiv + self.train_frame_size[1]/2/self.postdiv,
                            train_frame_center[1]/self.postdiv - self.train_frame_size[2]/2/self.postdiv:
                            train_frame_center[1]/self.postdiv + self.train_frame_size[2]/2/self.postdiv
                        ]   
                    
                    flow_mean = flow_frame.mean(axis=(0,1))/10.
                    flow_mean_norm = np.sum(flow_mean**2)**(1./2)
                    
                    dt = self.train_frame_size[0] + self.predict_interval + self.predict_frame_size[0] - 1 
                    dx = flow_mean[1] * dt * self.tdiv / self.prediv
                    dy = flow_mean[0] * dt * self.tdiv / self.prediv
                    track_frame_center = (train_frame_center[0] - int(round(dx)), train_frame_center[1] - int(round(dy)))                
                    if track_frame_center[0]-self.train_frame_size[1]/2 < 0 or \
                            track_frame_center[0]+self.train_frame_size[1]/2 > self.matrix.shape[2] or \
                            track_frame_center[1]-self.train_frame_size[2]/2 < 0 or \
                            track_frame_center[1]+self.train_frame_size[2]/2 > self.matrix.shape[3]:
                        continue
                    
                    if not self.track:
                        flow_mean *= 0.0   # track_frame_center = train_frame_center
                                
                    for j in range(track_frames.shape[0]):
                        dt = self.train_frame_size[0] + self.predict_interval + self.predict_frame_size[0] - 1 - j
                        dx = flow_mean[1] * dt * self.tdiv / self.prediv
                        dy = flow_mean[0] * dt * self.tdiv / self.prediv
                        track_frame_center = (train_frame_center[0] - int(round(dx)), train_frame_center[1] - int(round(dy)))
                        
                        track_frames[j] = self.matrix[month,
                                i-self.train_frame_size[0]-self.predict_interval-self.predict_frame_size[0]+1+j,
                                track_frame_center[0]-self.train_frame_size[1]/2:
                                track_frame_center[0]+self.train_frame_size[1]/2,
                                track_frame_center[1]-self.train_frame_size[2]/2:
                                track_frame_center[1]+self.train_frame_size[2]/2
                            ].astype('float')   
                    
                    ds = cv2.resize(track_frames.transpose((1,2,0)), (0,0), fx=1./self.postdiv, fy=1./self.postdiv).transpose((2,0,1))
                    x = ds.round().astype('uint8').flatten()
                    self.X[self.example_cnt % self.num_examples] = x
                    
                    if self.predict_style == 'interval':
                        rain = self.matrix[month, 
                                       i-self.predict_frame_size[0]+1:i+1, 
                                       predict_frame_center[0],
                                       predict_frame_center[1]
                                    ].max() >= self.threshold
                        self.y[self.example_cnt % self.num_examples, 0] = rain
                    else:
                        for j in range(self.predict_frame_size[0]):
                            rain = self.matrix[month,
                                               i - self.predict_frame_size[0] + 1 + j, 
                                               predict_frame_center[0], 
                                               predict_frame_center[1]] >= self.threshold
                            self.y[self.example_cnt % self.num_examples, j] = rain
                    self.example_cnt += 1

        t1 = time.time()
        print 'Done. Total:', self.example_cnt, 'Time:', t1 - t0
                
    def show_random_examples(self):
        h_center_low = self.train_frame_size[1]/2*self.showdiv
        h_center_high = self.matrix.shape[2] - self.train_frame_size[1]/2*self.showdiv
        
        w_center_low = self.train_frame_size[2]/2*self.showdiv
        w_center_high = self.matrix.shape[3] - self.train_frame_size[2]/2*self.showdiv
        
        pred_func, pred_func_track = self.build_pred_func()
        
        track_frames = np.zeros(self.train_frame_size)
        future_track_frames = np.zeros((self.predict_interval + self.predict_frame_size[0],
                                 self.train_frame_size[1],
                                 self.train_frame_size[2]))        
#        track_frames = np.zeros((self.train_frame_size[0] + self.predict_interval + self.predict_frame_size[0],
#                                 self.train_frame_size[1],
#                                 self.train_frame_size[2]))
        track_frames_ds = np.zeros_like(track_frames)
        future_track_frames_ds = np.zeros_like(future_track_frames)
        
        nrights = 0
        nrights_track = 0
        nrights_flow = 0
        nexamples = 0
        while nexamples < 100000:
            month = np.random.randint(self.matrix.shape[0])
            i = np.random.randint(self.train_frame_size[0] + self.predict_interval + self.predict_frame_size[0] - 1, 
                                  self.matrix.shape[1])
            if not self.usable(i):
                continue
            
            h_center = np.random.randint(h_center_low, h_center_high)
            w_center = np.random.randint(w_center_low, w_center_high)
            predict_frame_center = train_frame_center = (h_center, w_center)
            
            flow_frame = self.flow[month,
                    i-self.predict_interval-self.predict_frame_size[0],
                    train_frame_center[0]/self.postdiv - self.train_frame_size[1]/2/self.postdiv:
                    train_frame_center[0]/self.postdiv + self.train_frame_size[1]/2/self.postdiv,
                    train_frame_center[1]/self.postdiv - self.train_frame_size[2]/2/self.postdiv:
                    train_frame_center[1]/self.postdiv + self.train_frame_size[2]/2/self.postdiv
                ]   
            
            flow_mean = flow_frame.mean(axis=(0,1))/10.
            flow_mean_norm = np.sum(flow_mean**2)**(1./2)
            
            dt = self.train_frame_size[0] + self.predict_interval + self.predict_frame_size[0] - 1 
            dx = flow_mean[1] * dt * self.tdiv / self.prediv
            dy = flow_mean[0] * dt * self.tdiv / self.prediv
            track_frame_center = (train_frame_center[0] - int(round(dx)), train_frame_center[1] - int(round(dy)))                
            if track_frame_center[0]-self.train_frame_size[1]/2 < 0 or \
                    track_frame_center[0]+self.train_frame_size[1]/2 > self.matrix.shape[2] or \
                    track_frame_center[1]-self.train_frame_size[2]/2 < 0 or \
                    track_frame_center[1]+self.train_frame_size[2]/2 > self.matrix.shape[3]:
                continue

            train_frames = self.matrix[month,
                    i-self.train_frame_size[0]-self.predict_interval-self.predict_frame_size[0]+1:
                    i-self.predict_interval-self.predict_frame_size[0]+1,
                    train_frame_center[0]-self.train_frame_size[1]/2:
                    train_frame_center[0]+self.train_frame_size[1]/2,
                    train_frame_center[1]-self.train_frame_size[2]/2:
                    train_frame_center[1]+self.train_frame_size[2]/2
                ].astype('float')

            if np.sum(train_frames[-1] >= self.threshold) < self.pixnum_threshold:
                continue
                
            ds = cv2.resize(train_frames.transpose((1,2,0)), (0,0), fx=1./self.postdiv, fy=1./self.postdiv).transpose((2,0,1))
            x = ds.round().astype('uint8').flatten()
            rain_prob = pred_func(x.reshape(1, x.shape[0]))[0][0]
            pred_rain = rain_prob >= 0.5
                    
            for j in range(track_frames.shape[0]):
                dt = self.train_frame_size[0] + self.predict_interval + self.predict_frame_size[0] - 1 - j
                dx = flow_mean[1] * dt * self.tdiv / self.prediv
                dy = flow_mean[0] * dt * self.tdiv / self.prediv
                track_frame_center = (train_frame_center[0] - int(round(dx)), train_frame_center[1] - int(round(dy)))
                
                track_frames[j] = self.matrix[month,
                        i-self.train_frame_size[0]-self.predict_interval-self.predict_frame_size[0]+1+j,
                        track_frame_center[0]-self.train_frame_size[1]/2:
                        track_frame_center[0]+self.train_frame_size[1]/2,
                        track_frame_center[1]-self.train_frame_size[2]/2:
                        track_frame_center[1]+self.train_frame_size[2]/2
                    ].astype('float')    
                    
#            if np.sum(track_frames[-1] >= self.threshold) < self.pixnum_threshold:
#                continue
            
            ds = cv2.resize(track_frames.transpose((1,2,0)), (0,0), fx=1./self.postdiv, fy=1./self.postdiv).transpose((2,0,1))
            x = ds.round().astype('uint8').flatten()
            rain_prob_track = pred_func_track(x.reshape(1, x.shape[0]))[0][0]
            pred_rain_track = rain_prob_track >= 0.5
            
            pred_rain_flow = track_frames[-1, self.train_frame_size[1]/2, self.train_frame_size[2]/2] >= self.threshold
    
            rain = self.matrix[month, i, predict_frame_center[0], predict_frame_center[1]] >= self.threshold
            
            nexamples += 1
            if pred_rain == rain:
                nrights += 1
            if pred_rain_track == rain:
                nrights_track += 1
            if pred_rain_flow == rain:
                nrights_flow += 1
                
            if nexamples % 1000 == 0:
                print nexamples
            continue
                
            c0 = True if self.show_right is None else (pred_rain == rain) == self.show_right
            c1 = True if self.show_right_track is None else (pred_rain_track == rain) == self.show_right_track
            if c0 and c1:
#                print 'sum =', np.sum(train_frames[-1] >= self.threshold)
                print 'flow_mean =', flow_mean, flow_mean_norm
                print 'rain_prob =', rain_prob, 'rain_prob_track =', rain_prob_track
                print 'center_val =', self.matrix[month, i, predict_frame_center[0], predict_frame_center[1]]
                
                track_center_vals = np.concatenate((track_frames[:, self.train_frame_size[1]/2, self.train_frame_size[2]/2],
                                future_track_frames[:, self.train_frame_size[1]/2, self.train_frame_size[2]/2]))
                print 'track_center_vals =', track_center_vals
                
                future_frames = self.matrix[month,
                    i-self.predict_interval-self.predict_frame_size[0]+1 : i+1,
                    train_frame_center[0]-self.train_frame_size[1]/2:
                    train_frame_center[0]+self.train_frame_size[1]/2,
                    train_frame_center[1]-self.train_frame_size[2]/2:
                    train_frame_center[1]+self.train_frame_size[2]/2
                ].astype('float')                
                   
                train_frames_ds = self.matrix[month,
                        i-self.train_frame_size[0]-self.predict_interval-self.predict_frame_size[0]+1:
                        i-self.predict_interval-self.predict_frame_size[0]+1,
                        train_frame_center[0]-self.train_frame_size[1]/2*self.showdiv:
                        train_frame_center[0]+self.train_frame_size[1]/2*self.showdiv,
                        train_frame_center[1]-self.train_frame_size[2]/2*self.showdiv:
                        train_frame_center[1]+self.train_frame_size[2]/2*self.showdiv
                    ].astype('float')
                train_frames_ds = cv2.resize(train_frames_ds.transpose((1,2,0)), (0,0), 
                                        fx=1./self.showdiv, fy=1./self.showdiv).transpose((2,0,1))
    
                future_frames_ds = self.matrix[month,
                        i-self.predict_interval-self.predict_frame_size[0]+1 : i+1,
                        train_frame_center[0]-self.train_frame_size[1]/2*self.showdiv:
                        train_frame_center[0]+self.train_frame_size[1]/2*self.showdiv,
                        train_frame_center[1]-self.train_frame_size[2]/2*self.showdiv:
                        train_frame_center[1]+self.train_frame_size[2]/2*self.showdiv
                    ].astype('float')
                future_frames_ds = cv2.resize(future_frames_ds.transpose((1,2,0)), (0,0), 
                                        fx=1./self.showdiv, fy=1./self.showdiv).transpose((2,0,1))
                
                assert train_frames_ds.shape == train_frames.shape
                assert future_frames_ds.shape == future_frames.shape
                
                for j in range(future_track_frames.shape[0]):
                    dt = self.predict_interval + self.predict_frame_size[0] - 1 - j
                    dx = flow_mean[1] * dt * self.tdiv / self.prediv
                    dy = flow_mean[0] * dt * self.tdiv / self.prediv
                    track_frame_center = (train_frame_center[0] - int(round(dx)), train_frame_center[1] - int(round(dy)))
                    
                    future_track_frames[j] = self.matrix[month,
                            i-self.predict_interval-self.predict_frame_size[0]+1+j,
                            track_frame_center[0]-self.train_frame_size[1]/2:
                            track_frame_center[0]+self.train_frame_size[1]/2,
                            track_frame_center[1]-self.train_frame_size[2]/2:
                            track_frame_center[1]+self.train_frame_size[2]/2
                        ].astype('float')  
                          
                for j in range(track_frames.shape[0]):
                    dt = self.train_frame_size[0] + self.predict_interval + self.predict_frame_size[0] - 1 - j
                    dx = flow_mean[1] * dt * self.tdiv / self.prediv
                    dy = flow_mean[0] * dt * self.tdiv / self.prediv
                    track_frame_center = (train_frame_center[0] - int(round(dx)), train_frame_center[1] - int(round(dy)))
                    
                    track_frames_ds_j = self.matrix[month,
                            i-self.train_frame_size[0]-self.predict_interval-self.predict_frame_size[0]+1+j,
                            track_frame_center[0]-self.train_frame_size[1]/2*self.showdiv:
                            track_frame_center[0]+self.train_frame_size[1]/2*self.showdiv,
                            track_frame_center[1]-self.train_frame_size[2]/2*self.showdiv:
                            track_frame_center[1]+self.train_frame_size[2]/2*self.showdiv
                        ].astype('float')     
                    track_frames_ds[j] = cv2.resize(track_frames_ds_j, (0,0), 
                                        fx=1./self.showdiv, fy=1./self.showdiv)
                    
                for j in range(future_track_frames.shape[0]):
                    dt = self.predict_interval + self.predict_frame_size[0] - 1 - j
                    dx = flow_mean[1] * dt * self.tdiv / self.prediv
                    dy = flow_mean[0] * dt * self.tdiv / self.prediv
                    track_frame_center = (train_frame_center[0] - int(round(dx)), train_frame_center[1] - int(round(dy)))
                    
                    if track_frame_center[0]-self.train_frame_size[1]/2*self.showdiv < 0 or \
                            track_frame_center[0]+self.train_frame_size[1]/2*self.showdiv > self.matrix.shape[2] or \
                            track_frame_center[1]-self.train_frame_size[2]/2*self.showdiv < 0 or \
                            track_frame_center[1]+self.train_frame_size[2]/2*self.showdiv > self.matrix.shape[3]:
                        future_track_frames_ds[j] *= 0.0
                    else:
                        track_frames_ds_j = self.matrix[month,
                                i-self.predict_interval-self.predict_frame_size[0]+1+j,
                                track_frame_center[0]-self.train_frame_size[1]/2*self.showdiv:
                                track_frame_center[0]+self.train_frame_size[1]/2*self.showdiv,
                                track_frame_center[1]-self.train_frame_size[2]/2*self.showdiv:
                                track_frame_center[1]+self.train_frame_size[2]/2*self.showdiv
                            ].astype('float')     
                        future_track_frames_ds[j] = cv2.resize(track_frames_ds_j, (0,0), 
                                            fx=1./self.showdiv, fy=1./self.showdiv)
                
                pv = patch_viewer.PatchViewer(grid_shape=(4, track_frames.shape[0] + future_frames.shape[0]), 
                                              patch_shape=[train_frames.shape[1], train_frames.shape[2]])
                for fidx in range(train_frames.shape[0]):
                    pv.add_patch(train_frames[fidx], activation=0)
                for fidx in range(future_frames.shape[0]):
                    pv.add_patch(future_frames[fidx], activation=0)
                    
                for fidx in range(train_frames_ds.shape[0]):
                    pv.add_patch(train_frames_ds[fidx], activation=0)
                for fidx in range(future_frames_ds.shape[0]):
                    pv.add_patch(future_frames_ds[fidx], activation=0)                    
                    
                for fidx in range(track_frames.shape[0]):
                    pv.add_patch(track_frames[fidx], activation=0)
                for fidx in range(future_track_frames.shape[0]):
                    pv.add_patch(future_track_frames[fidx], activation=0)
                    
                for fidx in range(track_frames_ds.shape[0]):
                    pv.add_patch(track_frames_ds[fidx], activation=0)
                for fidx in range(future_track_frames_ds.shape[0]):
                    pv.add_patch(future_track_frames_ds[fidx], activation=0)
                    
                pv.show()
                
        print 'nrights =', nrights
        print 'nrights_track =', nrights_track
        print 'nrights_flow =', nrights_flow
                
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
            
class CLOUDFLOW(dense_design_matrix.DenseDesignMatrix):
    def __init__(self,  
                 which_set,
                 num_examples,
                 threshold = 3,
                 pixnum_threshold = 1,
                 prediv = 2,
                 postdiv = 2,
                 tdiv = 2,
                 train_frame_size = (3,30,30),
                 predict_frame_size = (1,1,1),
                 predict_interval = 2,
                 stride = (3,3),
                 tstride = 1,
                 tstart=0,
                 tstop=3600,
                 data_files = [],
                 axes=('c', 0, 1, 'b'),
                 examples_per_image = None,
                 video_shape = (7200, 477, 477),
                 image_border=(88, 88),
                 train_slot=50,   # 5 hours
                 valid_slot=20,   # 2 hours
                 test_slot=30,   # 3 hours
                 predict_style='point',
                 with_flow_feature=False
                 ):

        assert predict_style in ['interval', 'point']
        self.__dict__.update(locals())
        del self.self
        print '\nBuilding', which_set, 'set...'
        
        self.train_dim = self.train_frame_size[0]*\
                    (self.train_frame_size[1])/self.postdiv*\
                    (self.train_frame_size[2])/self.postdiv
        if self.with_flow_feature:
            self.flow_dim = 2* \
                        (self.train_frame_size[1])/self.postdiv* \
                        (self.train_frame_size[2])/self.postdiv
            self.train_dim += self.flow_dim
        predict_len = self.predict_frame_size[0] if predict_style == 'point' else 1
        self.predict_dim = predict_len * \
                    (self.predict_frame_size[1]) * \
                    (self.predict_frame_size[2])
        
        self.image_border = (np.ceil(image_border[0]/prediv), 
                             np.ceil(image_border[1]/prediv)) 

        print 'Preallocating X and y...',
#        example_interval = (self.train_frame_size[0] + self.predict_interval + 
#                            self.predict_frame_size[0] - 1)
#        num_images = (tstop - tstart - example_interval) / tstride
#        self.example_interval = example_interval

        X = np.zeros((num_examples, self.train_dim), dtype='uint8')
        y = np.zeros((num_examples, self.predict_dim), dtype='uint8')
        print 'done.' 

        shape = (self.train_frame_size[1] / self.postdiv,  #rows
                 self.train_frame_size[2] / self.postdiv,  #cols
                 self.train_frame_size[0]   #frames, i.e. channels
                 )     
        view_converter = dense_design_matrix.DefaultViewConverter(shape, self.axes)
        super(CLOUDFLOW,self).__init__(X = X, y = y, view_converter = view_converter)
        
        self.train_slot /= tdiv
        self.valid_slot /= tdiv
        self.test_slot /= tdiv
        self.whole_slot = self.train_slot + self.valid_slot + self.test_slot
        assert which_set in ['train', 'valid', 'test']
        if which_set == 'train':
            self.usable_start = 0
            self.usable_stop = self.train_slot - 1
        elif which_set == 'valid':
            self.usable_start = self.train_slot
            self.usable_stop = self.train_slot + self.valid_slot - 1
        else:
            self.usable_start = self.train_slot + self.valid_slot
            self.usable_stop = self.train_slot + self.valid_slot + self.test_slot - 1
                    
        data_dir = '/home/yuanxy/projects/nowcasting/data/'
        ds_shape = (np.ceil(video_shape[0]*1./tdiv), 
                    np.round(video_shape[1]*1./prediv), 
                    np.round(video_shape[2]*1./prediv))

        matrix = np.zeros(video_shape, dtype='uint8')
        flow = np.zeros((video_shape[0], 
                    np.round(video_shape[1]*1./prediv/postdiv), 
                    np.round(video_shape[2]*1./prediv/postdiv),
                    2))
        #ds = np.zeros((ds_shape[0]*tdiv, ds_shape[1], ds_shape[2]), dtype='uint8')
        ds = np.zeros((ds_shape[0]*tdiv, ds_shape[1], ds_shape[2]))
        
        self.example_cnt = 0
        self.pos_cnt = 0
        self.neg_cnt = 0
        for month in range(len(data_files)):
            data_file = data_files[month]
            print '\n',data_file
            ramdisk_root = '/mnt/ramdisk/'
            npy_file = ramdisk_root + data_file.replace('.pkl.gz', '.npy')
            npy_flow_file = ramdisk_root + data_file.replace('.pkl.gz', '_flow256.npy')
            if os.path.isfile(npy_file):
                print 'Cached. Loading data from ramdisk ...',
                self.matrix = np.load(npy_file)
                self.flow = np.load(npy_flow_file)
                print 'done.'
            else:
                print 'Loading data from disk and computing flow ...',
                t0 = time.time()
                f = gzip.open(data_dir + data_file)
                #matrix = np.zeros(video_shape, dtype='uint8')
                matrix *= 0
                flow *= 0.0
                for i in xrange(video_shape[0]):
                    try:
                        t,pix = cPickle.load(f)
                        pix = pix * (pix < 255)
                        #Gaussian_pix = cv2.GaussianBlur(pix,(5,5),sigmaX=0)               
                        #matrix[i] = Gaussian_pix
                        if "AZ9280" in data_file:  # for chengdu
                            border = [(pix.shape[0] - video_shape[1]) / 2,
                                      (pix.shape[1] - video_shape[2]) / 2]
                            pix = pix[border[0]:pix.shape[0]-border[0],
                                      border[1]:pix.shape[1]-border[1]]
                        matrix[i] = pix
                        
                        if i == 0:
                            continue
#                        flow = flow[image_border[0]:video_shape[1]-image_border[0],
#                                    image_border[1]:video_shape[2]-image_border[1],
#                                    :]
                        flow_i, _ = get_normalized_flow(matrix[i-1:i+1])
                                
                        # downsample inplace to save memory
                        flow[i] = cv2.resize(flow_i, (0, 0), fx=1./prediv/postdiv, fy=1./prediv/postdiv) 
                    except Exception, e:
                        traceback.print_exc()
                        break;
                f.close()
                self.matrix = matrix
                t1 = time.time()
                print 'done.', t1 - t0, 'seconds'     
            
                if prediv != 1:
                    print 'Downsampling ...',
                    ds *= 0.0
                    for i in range(video_shape[0]):
                        ds[i] = cv2.resize(matrix[i].astype('float'), (0, 0), fx=1./prediv, fy=1./prediv)
                    self.matrix = ds.reshape((ds_shape[0], tdiv, ds_shape[1], ds_shape[2])
                                    ).mean(axis=1).round().astype('uint8')

#                    flowds = flow # spatial downsampling has already been done
#                    self.flow = (flowds.reshape((flowds.shape[0]/tdiv, tdiv, flowds.shape[1], flowds.shape[2], flowds.shape[3])).mean(axis=1)
#                                 + 8.).clip(min=0.0).round().astype('uint8') 
                    flowds = flow.reshape((flow.shape[0]/tdiv, tdiv, flow.shape[1], flow.shape[2], flow.shape[3])).mean(axis=1)
                    maxval = 12.
                    minval = -12.
                    flowds[flowds > maxval] = maxval
                    flowds[flowds < minval] = minval
                    flowds = (flowds * 10.) + 128.  # in range [128-120, 128+120]
                    self.flow = flowds.round().astype('uint8')
                    print 'done.'
                
                print 'Caching data to ramdisk ...',
                np.save(npy_file, self.matrix)
                np.save(npy_flow_file, self.flow)
                print 'done.'
                
            self.gen_random_examples()
        
        assert not np.any(np.isnan(self.X))
        
        #pixel_val = self.X[:, 0 : self.train_dim-self.flow_dim]
        #flow_val = self.X[:,self.train_dim-self.flow_dim : self.train_dim]
        #print 'pixel max, mean, min:', pixel_val.max(), pixel_val.mean(), pixel_val.min()
#        print 'flow max, mean, min:', flow_val.max(), flow_val.mean(), flow_val.min()
#        print 'sorting flow_val...'
#        import pylab as plt
#        np.sort(flow_val.flatten())
#        plt.plot()
#        plt.show()
        
    def usable(self, pos):
        start = pos-self.train_frame_size[0]-self.predict_interval - self.predict_frame_size[0] + 1
        stop = pos
        if start % self.whole_slot >= self.usable_start and \
            stop % self.whole_slot <= self.usable_stop: 
            return True
        else:
            return False
    
    def gen_random_examples(self):
        h_center_low = self.image_border[0] + self.train_frame_size[1]/2
        h_center_high = self.matrix.shape[1] - self.image_border[0] - self.train_frame_size[1]/2
        assert h_center_low < h_center_high
        
        w_center_low = self.image_border[1] + self.train_frame_size[2]/2
        w_center_high = self.matrix.shape[2] - self.image_border[1] - self.train_frame_size[2]/2
        assert w_center_low < w_center_high

        print 'Generating random examples ...',
        t0 = time.time()
        
        ds2 = np.zeros((self.train_frame_size[0],
                        np.round(self.train_frame_size[1]*1./self.postdiv),
                        np.round(self.train_frame_size[2]*1./self.postdiv)))
        for i in range(self.tstart + self.train_frame_size[0] + 
                       self.predict_interval + self.predict_frame_size[0] - 1, 
                       self.tstop, 
                       self.tstride):
            if not self.usable(i):
                continue
            for _ in range(self.examples_per_image):
                h_center = np.random.randint(h_center_low, h_center_high)
                w_center = np.random.randint(w_center_low, w_center_high)
                predict_frame_center = train_frame_center = (h_center, w_center)

                train_frames = self.matrix[
                        i-self.train_frame_size[0]-self.predict_interval:
                        i-self.predict_interval,
                        train_frame_center[0]-self.train_frame_size[1]/2:
                        train_frame_center[0]+self.train_frame_size[1]/2,
                        train_frame_center[1]-self.train_frame_size[2]/2:
                        train_frame_center[1]+self.train_frame_size[2]/2
                    ]

                if np.sum(train_frames[-1] >= self.threshold) < self.pixnum_threshold:
                    continue;

                if self.postdiv !=1:
                    for j in range(self.train_frame_size[0]):  
                        ds2[j] = cv2.resize(train_frames[j].astype('float'), 
                                        (0, 0), 
                                        fx=1./self.postdiv, 
                                        fy=1./self.postdiv)

                    #del train_frames
                    train_frames = ds2.round().astype('uint8')
                
                x = train_frames.flatten()
                if self.with_flow_feature:
                    flow_frame = self.flow[i-self.predict_interval-1,
                        train_frame_center[0]/2-self.train_frame_size[1]/4:
                        train_frame_center[0]/2+self.train_frame_size[1]/4,
                        train_frame_center[1]/2-self.train_frame_size[2]/4:
                        train_frame_center[1]/2+self.train_frame_size[2]/4
                     ]   
                    x = np.concatenate([x, flow_frame.flatten()])                
                self.X[self.example_cnt % self.num_examples] = x
                
                if self.predict_style == 'interval':
                    rain = self.matrix[i-self.predict_frame_size[0]+1:i+1, 
                                   predict_frame_center[0],
                                   predict_frame_center[1]
                                ].max() >= self.threshold
                    self.y[self.example_cnt % self.num_examples, 0] = rain
                else:
                    for j in range(self.predict_frame_size[0]):
                        rain = self.matrix[i - self.predict_frame_size[0] + 1 + j, 
                                           predict_frame_center[0], 
                                           predict_frame_center[1]] >= self.threshold
                        self.y[self.example_cnt % self.num_examples, j] = rain
                    #self.pos_cnt += rain
                self.example_cnt += 1

        t1 = time.time()

        print 'done. Total:', self.example_cnt
#        print 'time =', t1 - t0


class CLOUDFLY(dense_design_matrix.DenseDesignMatrix):
    def __init__(self,  
                 which_set,
                 model_kind,
                 num_examples,
                 threshold,
                 train_pixnum_threshold = 12,
                 test_pixnum_threshold = 2,
                 only_1Model = False,
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
                 tstop=3000,
                 data_files = [],
                 axes=('c', 0, 1, 'b'),
                 examples_per_image = None,
                 video_shape = (6000, 477, 477),
                 image_border=(90, 90),
                 train_slot=50,   # 5 hours
                 valid_slot=20,   # 2 hours
                 test_slot=30,   # 3 hours
                 predict_style='point'
                 ):

        assert predict_style in ['interval', 'point']
        self.__dict__.update(locals())
        del self.self
        
        
        self.train_dim = self.train_frame_size[0]*\
                    (self.train_frame_size[1])/self.postdiv*\
                    (self.train_frame_size[2])/self.postdiv
        predict_len = self.predict_frame_size[0] if predict_style == 'point' else 1
        self.predict_dim = predict_len * \
                    (self.predict_frame_size[1]) * \
                    (self.predict_frame_size[2])
        
        self.image_border = (np.ceil(image_border[0]/prediv), 
                             np.ceil(image_border[1]/prediv)) 

        print 'Preallocating X and y...'
#        example_interval = (self.train_frame_size[0] + self.predict_interval + 
#                            self.predict_frame_size[0] - 1)
#        num_images = (tstop - tstart - example_interval) / tstride
#        self.example_interval = example_interval

        X = np.zeros((num_examples, self.train_dim), dtype='uint8')
        y = np.zeros((num_examples, self.predict_dim), dtype='uint8')
        print 'Preallocating X and y done.' 

        shape = (self.train_frame_size[1] / self.postdiv,  #rows
                 self.train_frame_size[2] / self.postdiv,  #cols
                 self.train_frame_size[0]   #frames, i.e. channels
                 )     
        view_converter = dense_design_matrix.DefaultViewConverter(shape, self.axes)
        super(CLOUDFLY,self).__init__(X = X, y = y, view_converter = view_converter)
        
        self.train_slot /= tdiv
        self.valid_slot /= tdiv
        self.test_slot /= tdiv
        self.whole_slot = self.train_slot + self.valid_slot + self.test_slot
        assert which_set in ['train', 'valid', 'test']
        if which_set == 'train':
            self.usable_start = 0
            self.usable_stop = self.train_slot - 1
        elif which_set == 'valid':
            self.usable_start = self.train_slot
            self.usable_stop = self.train_slot + self.valid_slot - 1
        else:
            self.usable_start = self.train_slot + self.valid_slot
            self.usable_stop = self.train_slot + self.valid_slot + self.test_slot - 1
                    
        data_dir = '/home/yuanxy/projects/nowcasting/data/'
        ds_shape = (np.ceil(video_shape[0]*1./tdiv), 
                    np.round(video_shape[1]*1./prediv), 
                    np.round(video_shape[2]*1./prediv))

#        downsampled = np.zeros((ds_shape[0]*len(data_files), 
#                                ds_shape[1],
#                                ds_shape[2]), dtype='uint8')
#        print downsampled.shape

        #nframes = 0
        matrix = np.zeros(video_shape, dtype='uint8')
        #ds = np.zeros((ds_shape[0]*tdiv, ds_shape[1], ds_shape[2]), dtype='uint8')
        ds = np.zeros((ds_shape[0]*tdiv, ds_shape[1], ds_shape[2]))
        self.example_cnt = 0
        self.pos_cnt = 0
        self.neg_cnt = 0
        for month in range(len(data_files)):
            data_file = data_files[month]
            print '\n',data_file
            ramdisk_root = '/mnt/ramdisk/'
            npy_file = ramdisk_root + data_file.replace('.pkl.gz', '.npy')
            if os.path.isfile(npy_file):
                print 'Cached. Loading data from ramdisk...'
                self.matrix = np.load(npy_file)
                print 'done.'
            else:
                print 'Loading data from disk...'
                f = gzip.open(data_dir + data_file)
                #matrix = np.zeros(video_shape, dtype='uint8')
                matrix *= 0
                for i in xrange(video_shape[0]):
                    try:
                        t,pix = cPickle.load(f)
                        pix = pix * (pix < 255)
                        #Gaussian_pix = cv2.GaussianBlur(pix,(5,5),sigmaX=0)               
                        #matrix[i] = Gaussian_pix
                        if "AZ9280" in data_file:  # for chengdu
                            border = [(pix.shape[0] - video_shape[1]) / 2,
                                      (pix.shape[1] - video_shape[2]) / 2]
                            pix = pix[border[0]:pix.shape[0]-border[0],
                                      border[1]:pix.shape[1]-border[1]]
                        matrix[i] = pix
                    except:
                        break;
                f.close()
                self.matrix = matrix
                print 'done.'     
            
                if prediv != 1:
                    print 'Downsampling...'
                    ds *= 0
                    for i in range(video_shape[0]):
                        ds[i] = cv2.resize(matrix[i].astype('float'), 
                                   (0, 0), 
                                   fx=1./prediv, 
                                   fy=1./prediv)
                    self.matrix = ds.reshape((ds_shape[0], 
                                     tdiv, 
                                     ds_shape[1],
                                     ds_shape[2])
                                    ).mean(axis=1).round().astype('uint8')
                    print 'done.'
                
                print 'Caching data to ramdisk...'
                np.save(npy_file, self.matrix)
                print 'done.'
                
            self.gen_random_examples()
        
        assert not np.any(np.isnan(self.X))
       
    def usable(self, pos):
        start = pos-self.train_frame_size[0]-self.predict_interval - self.predict_frame_size[0] + 1
        stop = pos
        if start % self.whole_slot >= self.usable_start and \
            stop % self.whole_slot <= self.usable_stop: 
            return True
        else:
            return False
            
    def gen_random_examples(self):

        train_frame_radius = ((self.train_frame_size[0] - 1.)/2,
                              (self.train_frame_size[1])/2, 
                              (self.train_frame_size[2])/2)
        predict_frame_radius = ((self.predict_frame_size[0] - 1.)/2,
                              (self.predict_frame_size[1])/2, 
                              (self.predict_frame_size[2])/2)
        h_center_low = self.image_border[0]
        h_center_high = self.matrix.shape[1] - self.image_border[0]
        assert h_center_low < h_center_high
        
        w_center_low = self.image_border[1]
        w_center_high = self.matrix.shape[2] - self.image_border[1]
        assert w_center_low < w_center_high

        print 'Generating random examples...'
        t0 = time.time()
        #X1 = []
        #X2 = []
        #self.X *= 0
        #self.y *= 0

        #pos = 0
        #neg = 0
        #nexamples = 0

#        new_train_frames = np.zeros((self.train_frame_size[0],
#                                     self.train_frame_size[1]/self.postdiv,
#                                     self.train_frame_size[2]/self.postdiv), dtype='uint8')
        ds2 = np.zeros((self.train_frame_size[0],
                        np.round(self.train_frame_size[1]*1./self.postdiv),
                        np.round(self.train_frame_size[2]*1./self.postdiv)))
        for i in range(self.tstart + self.train_frame_size[0] + 
                       self.predict_interval + self.predict_frame_size[0] - 1, 
                       self.tstop, 
                       self.tstride):
            if not self.usable(i):
                continue
            for _ in range(self.examples_per_image):
                h_center = np.random.randint(h_center_low, h_center_high)
                w_center = np.random.randint(w_center_low, w_center_high)
                predict_frame_center = train_frame_center = (h_center, w_center)

                train_frames = self.matrix[
                        i-self.train_frame_size[0]-self.predict_interval:
                        i-self.predict_interval,
                        train_frame_center[0]-train_frame_radius[1]:
                        train_frame_center[0]+train_frame_radius[1],
                        train_frame_center[1]-train_frame_radius[2]:
                        train_frame_center[1]+train_frame_radius[2]
                    ]

                point_kind = self.classify_point(train_frames[-1],
                                            self.only_1Model) 
                if point_kind != self.model_kind:
                    continue;

                if self.postdiv !=1:
                    for j in range(self.train_frame_size[0]):  
                        ds2[j] = cv2.resize(train_frames[j].astype('float'), 
                                        (0, 0), 
                                        fx=1./self.postdiv, 
                                        fy=1./self.postdiv)

                    #del train_frames
                    train_frames = np.around(ds2).astype('uint8')
                    
                self.X[self.example_cnt % self.num_examples] = train_frames.flatten()
                
                if self.predict_style == 'interval':
                    rain = self.matrix[i-self.predict_frame_size[0]+1:i+1, 
                                   predict_frame_center[0],
                                   predict_frame_center[1]
                                ].max() >= self.threshold
                    self.y[self.example_cnt % self.num_examples, 0] = rain
                else:
                    for j in range(self.predict_frame_size[0]):
                        rain = self.matrix[i - self.predict_frame_size[0] + 1 + j, 
                                           predict_frame_center[0], 
                                           predict_frame_center[1]] >= self.threshold
                        self.y[self.example_cnt % self.num_examples, j] = rain
                    #self.pos_cnt += rain
                self.example_cnt += 1

        t1 = time.time()

        print 'Generating random examples done.'
        print 'pos:', self.pos_cnt, 'total:', self.example_cnt
        print 'time =', t1 - t0
        
    def gen_examples(self):
        train_frame_radius = ((self.train_frame_size[0] - 1.)/2,
                              (self.train_frame_size[1])/2, 
                              (self.train_frame_size[2])/2)
        predict_frame_radius = ((self.predict_frame_size[0] - 1.)/2,
                              (self.predict_frame_size[1])/2, 
                              (self.predict_frame_size[2])/2)
        
        h_center_low = self.image_border[0]
        h_center_high = self.matrix.shape[1] - self.image_border[0]
        assert h_center_low < h_center_high
        
        w_center_low = self.image_border[1]
        w_center_high = self.matrix.shape[2] - self.image_border[1]
        assert w_center_low < w_center_high

        print 'Generating examples...'
        t0 = time.time()
        pos = 0
        neg = 0
        nexamples = 0

        new_train_frames = np.zeros((self.train_frame_size[0],
                                     self.train_frame_size[1]/self.postdiv,
                                     self.train_frame_size[2]/self.postdiv))
        
        for i in range(self.tstart + self.example_interval, 
                       self.tstop, 
                       self.tstride):
            for h_center in np.arange(h_center_low, h_center_high, self.stride[0]):
                for w_center in np.arange(w_center_low, w_center_high, self.stride[1]):
                    predict_frame_center = train_frame_center = (h_center, w_center)

                    train_frames = self.matrix[
                        i-self.train_frame_size[0]-self.predict_interval:
                        i-self.predict_interval,
                        train_frame_center[0]-train_frame_radius[1]:
                        train_frame_center[0]+train_frame_radius[1],
                        train_frame_center[1]-train_frame_radius[2]:
                        train_frame_center[1]+train_frame_radius[2]
                    ]

                    point_kind = self.classify_point(train_frames[-1], self.only_1Model) 
                    if point_kind != self.model_kind:
                        continue;
                    
                    if self.postdiv !=1:
                        for j in range(self.train_frame_size[0]):  
                            ds = cv2.resize(train_frames[j].astype('float'), 
                                            (0, 0), 
                                            fx=1./self.postdiv, 
                                            fy=1./self.postdiv)
                            new_train_frames[j]= np.around(ds).astype('uint8')
    
                        del train_frames
                        train_frames = new_train_frames
                        
                    self.X[nexamples] = train_frames.flatten()
                    if self.matrix[i-self.predict_interval:i+1, 
                                   predict_frame_center[0],
                                   predict_frame_center[1]
                                ].max() >= self.threshold:
                        self.y[nexamples] = [1,]    # rain
                        pos += 1
                    else:
                        self.y[nexamples] = [0,]    # not rain
                        neg += 1
                    nexamples += 1
                                        
        t1 = time.time()
        print 'Generating examples done.'
        print 'pos:',pos,'neg:',neg, 'total:', pos + neg
        print 'time =', t1 - t0

    def classify_point(self,small_frame,oneModel):

        if oneModel:

            if np.sum(small_frame >= self.threshold) >= self.onemodel_pixnum_threshold:
                return 3
            else:
                return -1
        else:

            center = (np.floor(self.train_frame_size[1]/2.),np.floor(self.train_frame_size[2]/2.))

            if small_frame[center[0],center[1]] < self.threshold:
           
                if self.which_set != 'test':
                    if np.sum(small_frame > 3) > self.train_pixnum_threshold:
                        return 1
                    else:
                        return -1
                else:               
                    if np.sum(small_frame > 3) > self.test_pixnum_threshold:
                        return 1
                    else:
                        return -1
            else:
                return 2 
    

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
    return flow, flow_mean

# from api.py
MIN_FLOW_NORM=2
  
def flow_normlize(flow, flow_mean):
    flow_norm = numpy.linalg.norm(flow);
    flow_mean_norm = numpy.linalg.norm(flow_mean);
    
    if 0 < flow_norm < MIN_FLOW_NORM and flow_mean_norm > MIN_FLOW_NORM:
        flow = flow * flow_mean_norm / flow_norm
    
    return flow

# from opt_flow.py
def draw_flow(img, flow, step):
    h, w = img.shape[:2]
    y, x = map(np.ravel, np.mgrid[step/2:h:step, step/2:w:step])
    f = flow[y,x]
    x1 = x + f[:,0]
    y1 = y + f[:,1]
    lines = np.int32( np.vstack([x, y, x1, y1]).T )
    vis = cv2.cvtColor(img, cv.CV_GRAY2BGR)
    #print lines
    #cv2.polylines(vis, lines, 0, (0, 255, 0))
    for x_, y_, x1_, y1_ in np.int32(zip(x, y, x1, y1)):
        cv2.line(vis, (x_, y_), (x1_, y1_), (0, 255, 0))
    return vis

"""
while True:
    ret, img = cam.read()
    #img = cv2.pyrDown(img)
    gray = cv2.cvtColor(img, cv.CV_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray
    
    cv2.imshow('flow', draw_flow(gray, flow, 16))
    if cv2.waitKey(5) == 27:
       break
"""
import cPickle
import gzip
import pylab as plt
data_dir = '/home/yuanxy/projects/nowcasting/data/'

def show_flow_trend(data_files, 
                    video_shape = (7200, 477, 477), 
                    flow_trend_shape = (120, 30, 2),
                    out=None):    
    
    matrix = np.zeros(video_shape, dtype='uint8')
    
    hsv = np.zeros((flow_trend_shape[0], flow_trend_shape[1], 3), dtype='uint8')
    hsv[...,1] = 255

    rgbs = []
    for month in range(len(data_files)):
        print 'Loading data...'
        f = gzip.open(data_dir + data_files[month])
        #matrix = np.zeros(video_shape, dtype='uint8')
        matrix *= 0
        #flow_means *= 0.0
        for i in xrange(video_shape[0]):
            try:
                t,pix = cPickle.load(f)
                pix = pix * (pix < 255)
                #Gaussian_pix = cv2.GaussianBlur(pix,(5,5),sigmaX=0)               
                #matrix[i] = Gaussian_pix
                matrix[i] = pix
            except:
                break;
        f.close()
        print 'Loading data done.', i, 'frames loaded.'
        
        print 'Downsampling...'
        tdiv = 2
        prediv = 2
        ds_shape = (np.ceil(video_shape[0]*1./tdiv), 
                    np.round(video_shape[1]*1./prediv), 
                    np.round(video_shape[2]*1./prediv))

#        downsampled = np.zeros((ds_shape[0]*1, 
#                                ds_shape[1],
#                                ds_shape[2]), dtype='uint8')        
        ds = np.zeros((ds_shape[0]*tdiv, 
                       ds_shape[1], 
                       ds_shape[2]), dtype='uint8')
        for i in range(video_shape[0]):
            ds[i] = cv2.resize(matrix[i].astype('float'), 
                       (0, 0), 
                       fx=1./prediv, 
                       fy=1./prediv)
        ds = ds.reshape((ds_shape[0], 
                         tdiv, 
                         ds_shape[1],
                         ds_shape[2])
                        ).mean(axis=1)
        downsampled = np.around(ds).astype('uint8')
        print 'Downsampling done.'
        
        #del matrix
        #matrix = downsampled
                
        print 'Computing flow...'
        flow_means = np.zeros((downsampled.shape[0], 2))
        for i in xrange(1, downsampled.shape[0]):
            flow, flow_mean = getflow(downsampled[i-1:i+1])
            flow_means[i] = flow_mean
        print 'Computing flow done.'    
        
        print 'Rendering flow trend...'
        flow_trend = flow_means.reshape(flow_trend_shape)
        #mag, ang = cv2.cartToPolar(flow_trend[...,0], flow_trend[...,1])
        mag, ang = cv2.cartToPolar(-flow_trend[...,0], flow_trend[...,1]) # fix bug in opencv sample code 
        # hsv[...,0] = ang*180/np.pi/2
        hsv[...,0] = ang*255./np.pi/2  # fix bug in opencv sample code  
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        print 'Rendering flow trend done.'
    
        #cv2.imshow(rgb)
        #plt.imshow(rgb)
                
        rgbs.append(rgb)
        
    for i in range(len(data_files)):
        plt.subplot(1, len(data_files), i+1)
        plt.imshow(rgbs[i])
    if out is None:
        plt.show()
    else:
        plt.savefig(out, bbox_inches='tight')        
    return rgbs

#data_files = ['radar_img_matrix_AZ9290_201407_uint8.pkl.gz',
#              'radar_img_matrix_AZ9290_201408_uint8.pkl.gz',
#              'radar_img_matrix_AZ9290_201409_uint8.pkl.gz'
#              ]
#show_flow_trend(data_files=data_files, out='flowevolve_xian_201407-201409.png')

#data_files = ['radar_img_matrix_AZ9280_201407_uint8.pkl.gz',
#              'radar_img_matrix_AZ9280_201408_uint8.pkl.gz',
#              'radar_img_matrix_AZ9280_201409_uint8.pkl.gz'
#              ]
#show_flow_trend(data_files=data_files, 
#                video_shape = (7200, 509, 509), 
#                out='flowevolve_chengdu_201407-201409.png')

#data_files = ['radar_img_matrix_AZ9912_201407_uint8.pkl.gz',
#              'radar_img_matrix_AZ9912_201408_uint8.pkl.gz',
#              'radar_img_matrix_AZ9912_201409_uint8.pkl.gz'
#              ]
#show_flow_trend(data_files=data_files, out='flowevolve_yulin_201407-201409.png')
#
#data_files = ['radar_img_matrix_AZ9210_201408_uint8.pkl.gz',
#              'radar_img_matrix_AZ9210_201409_uint8.pkl.gz'
#              ]
#show_flow_trend(data_files=data_files, out='flowevolve_shanghai_201408-201409.png')
#
#data_files = ['radar_img_matrix_AZ9002_201407_uint8.pkl.gz',
#              'radar_img_matrix_AZ9002_201408_uint8.pkl.gz',
#              'radar_img_matrix_AZ9002_201409_uint8.pkl.gz'
#              ]
#show_flow_trend(data_files=data_files, out='flowevolve_qingpu_201407-201409.png')
#
#data_files = ['radar_img_matrix_AZ9200_201406_uint8.pkl.gz',
#              'radar_img_matrix_AZ9200_201407_uint8.pkl.gz',
#              'radar_img_matrix_AZ9200_201408_uint8.pkl.gz',
#              'radar_img_matrix_AZ9200_201409_uint8.pkl.gz'
#              ]
#show_flow_trend(data_files=data_files, out='flowevolve_guangzhou_201406-201409.png')
"""
video_shape = (7200, 477, 477)
flow_trend_shape = (120, 30, 2)

matrix = np.zeros(video_shape, dtype='uint8')

print 'Loading data...'
f = gzip.open(data_dir + 'radar_img_matrix_AZ9200_201408_uint8.pkl.gz')
#matrix = np.zeros(video_shape, dtype='uint8')
matrix *= 0
#flow_means *= 0.0
hsv = np.zeros((flow_trend_shape[0], flow_trend_shape[1], 3), dtype='uint8')
hsv[...,1] = 255

for i in xrange(video_shape[0]):
    try:
        t,pix = cPickle.load(f)
        pix = pix * (pix < 255)
        #Gaussian_pix = cv2.GaussianBlur(pix,(5,5),sigmaX=0)               
        #matrix[i] = Gaussian_pix
        matrix[i] = pix
    except:
        break;
f.close()
print 'Loading data done.'

print 'Downsampling...'
tdiv = 2
prediv = 2
ds_shape = (np.ceil(video_shape[0]*1./tdiv), 
            np.round(video_shape[1]*1./prediv), 
            np.round(video_shape[2]*1./prediv))

#        downsampled = np.zeros((ds_shape[0]*1, 
#                                ds_shape[1],
#                                ds_shape[2]), dtype='uint8')        
ds = np.zeros((ds_shape[0]*tdiv, 
               ds_shape[1], 
               ds_shape[2]), dtype='uint8')

for i in range(video_shape[0]):
    ds[i] = cv2.resize(matrix[i].astype('float'), 
               (0, 0), 
               fx=1./prediv, 
               fy=1./prediv)
ds = ds.reshape((ds_shape[0], 
                 tdiv, 
                 ds_shape[1],
                 ds_shape[2])
                ).mean(axis=1)
ds = np.around(ds).astype('uint8')
print 'Downsampling done.'

print 'Computing flow...'
flow_means = np.zeros((ds.shape[0], 2))
for i in xrange(1, ds.shape[0]):
    flow, flow_mean = getflow(ds[i-1:i+1])
    flow_means[i] = flow_mean
print 'Computing flow done.'    


print 'Rendering flow trend...'
flow_trend = flow_means.reshape(flow_trend_shape)
mag, ang = cv2.cartToPolar(-flow_trend[...,0], flow_trend[...,1])
hsv[...,0] = ang*255./np.pi/2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
print 'Rendering flow trend done.'

w = 30
for i in range(100*w, 110*w, 4):
    print flow_means[i]
    plt.imshow(ds[i])
    plt.show()
"""