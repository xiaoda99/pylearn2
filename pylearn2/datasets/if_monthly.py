"""
The MNIST dataset.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import cPickle

import numpy as N
np = N
from theano.compat.six.moves import xrange
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import control
from pylearn2.datasets import cache
from pylearn2.utils import serial
from pylearn2.utils.mnist_ubyte import read_mnist_images
from pylearn2.utils.mnist_ubyte import read_mnist_labels
from pylearn2.utils.rng import make_np_rng

def remove_extreme(s, cutoff):
    sorted = np.sort(s)
    cutoff_len = sorted.size * cutoff
    if sorted[0] * sorted[-1] < 0:
        low = sorted[cutoff_len]    
        s[s < low] = low
    high = sorted[-cutoff_len]
    s[s > high] = high
    return s

class IFMonthly(dense_design_matrix.DenseDesignMatrix):
    X = None
    y = None
    def __init__(self, which_set=None, start=None, stop=None, hist_len=1, examples_per_day=32402):
        self.hist_len = hist_len
        self.examples_per_day = examples_per_day
        cutoff = .001
        
        if IFMonthly.X is None:
            f = open('/home/xd/data/trading/IF1503_21f.pkl')
            print 'Loading...'
            X, y = cPickle.load(f)
            print 'Done.'
            f.close()
            
            for i in range(X.shape[1]):
                remove_extreme(X[:,i], cutoff)
            for i in range(y.shape[1]):
                remove_extreme(y[:,i], cutoff)
                
#            X_train = X[:X.shape[0]*2/3]
            X_train = X
            X_mean = X_train.mean(axis=0)
            X_train = X_train - X_mean
            X_stdev = np.sqrt(X_train.var(axis=0))
            X = X - X_mean
            X /= X_stdev  # Does not make a copy.
            
#            feature_mask = np.array([
#              # price indicators, less useful than volume indicators
#              1, #ask_price_delta,
#              1, #bid_price_delta,
#              1, #ask_price_delta_MA[4],
#              1, #bid_price_delta_MA[4],
#              #1, #ask_price_bias[20],
#              #1, #bid_price_bias[20],
#              1, #price_diff,
#              1, #price_diff_delta,
#              # volume indicators
#              1, #ask_vol,
#              1, #bid_vol,
#              1, #ask_vol2,
#              1, #bid_vol2,
#              1, #vol_diff,  # roughly the same useful as ask_vol combined with bid_vol
#              1, #vol_diff2,
#              1, #vol_diff_delta,  # less useful than vol_diff
#              1, #vol_diff_MA[5],  # little use
#              1, #deal_vol,
#              # other indicators
#              0, #pos_delta,
#              0, #pos_delta_MA[10]
#              ])
#            X *= feature_mask
            
            print 'X.shape, y.shape =', X.shape, y.shape
            X_hist = np.zeros((X.shape[0], X.shape[1], self.hist_len), dtype='float32')
            for shift in range(self.hist_len):
                X_hist[:,:,shift] = np.roll(X, shift, axis=0)
                X_hist[:shift,:,shift] = 0.
            IFMonthly.X = X_hist.reshape((X_hist.shape[0], X_hist.shape[1] * X_hist.shape[2]))
            IFMonthly.y = y.astype('float32')
            print 'X.shape =', X.shape

        super(IFMonthly, self).__init__(X=IFMonthly.X, y=IFMonthly.y)

        if which_set is not None:
            if which_set == 'train':
                start = 0
                stop = self.X.shape[0]*2/3
            else:
                assert which_set == 'test'
                start = self.X.shape[0]*2/3
                stop = self.X.shape[0]
        else:
            assert start is not None
            assert stop is not None

        self.X = self.X[start:stop, :]
        if len(self.y.shape) > 1:
            self.y = self.y[start:stop, :]
        else:
            self.y = self.y[start:stop]
        print 'Final: X.shape, y.shape =', self.X.shape, self.y.shape

def form_history(X, hist_len, hist_step=1):
    X_hist = np.zeros((X.shape[0], hist_len, X.shape[1]), dtype='float32')
    for i in range(hist_len):
        shift = i * hist_step
        X_hist[:,i,:] = np.roll(X, shift, axis=0)
        X_hist[:shift,i,:] = 0.
    return X_hist.reshape((X_hist.shape[0], X_hist.shape[1] * X_hist.shape[2]))
            
class IFMonthlyGain(dense_design_matrix.DenseDesignMatrix):
    X = None
    y = None
    def __init__(self, discount, direction, sight, which_set=None, start=None, stop=None, hist_len=1, examples_per_day=32402):
        self.hist_len = hist_len
        self.examples_per_day = examples_per_day
        self.cutoff = .001
        
        if IFMonthlyGain.X is None:
            discount_str = '_discount0' if not discount else ''
            f = open('/home/xd/data/trading/IF1503_gain_feat49%s.pkl' % (discount_str,))
            print 'Loading...'
            X, y = cPickle.load(f)
            print 'Done.'
            f.close()
            
            feature_mask = np.array([
              # price indicators, less useful than volume indicators
              1, #ask_price_delta[20],
              1, #bid_price_delta[20],
              1, #ask_price_delta_MA[4],
              1, #bid_price_delta_MA[4],
              #1, #ask_price_bias[20],
              #1, #bid_price_bias[20],
              1, #price_diff,
              1, #price_diff_delta,
              # volume indicators
              1, #ask_vol,
              1, #bid_vol,
              1, #ask_vol2,
              1, #bid_vol2,
              1, #vol_diff,  # roughly the same useful as ask_vol combined with bid_vol
              1, #vol_diff2,
              1, #vol_diff_delta,  # less useful than vol_diff
              1, #vol_diff_MA[5],  # little use
              1, #deal_vol,
              # other indicators
              1, #pos_delta,
              1, #pos_delta_MA[10]
              ])
#            X *= feature_mask
            
            print 'X.shape, y.shape =', X.shape, y.shape
            
            for i in range(X.shape[1]):
                remove_extreme(X[:,i], self.cutoff)
            for i in range(y.shape[1]):
                remove_extreme(y[:,i], self.cutoff)
                
            X_abs_max = np.abs(X).max(axis=0, keepdims=True)
#            with open('/home/xd/data/trading/IF1503_feat49_X_abs_max.pkl', 'wb') as f:
#                cPickle.dump(X_abs_max[0], f)
                
            X /= X_abs_max
            for i in range(0, X.shape[1]):
                print X[:, i].max(), X[:, i].min(), np.abs(X[:, i]).mean()
 
            IFMonthlyGain.X = form_history(X, self.hist_len)
            print 'IFMonthlyGain.X.shape =', IFMonthlyGain.X.shape
            
            y_index = direction * 4 + sight - 1
            IFMonthlyGain.y = y[:, y_index : y_index + 1].astype('float32')

        super(IFMonthlyGain, self).__init__(X=IFMonthlyGain.X, y=IFMonthlyGain.y)
        
        assert self.X.shape[0] % self.examples_per_day == 0
        self.n_days = self.X.shape[0] / self.examples_per_day
        
        n_training_days = self.n_days * 2 / 3
        n_test_days = self.n_days - n_training_days
        print 'n_days, n_training_days, n_test_days =', self.n_days, n_training_days, n_test_days
        if which_set is not None:
            if which_set == 'train':
                start = 0
                stop = n_training_days * self.examples_per_day
            else:
                assert which_set == 'test'
                start = n_training_days * self.examples_per_day
                stop = self.n_days * self.examples_per_day
        else:
            assert start is not None
            assert stop is not None

        self.X = self.X[start:stop, :]
        if self.X.shape[0] != stop - start:
            raise ValueError("X.shape[0]: %d. start: %d stop: %d"
                             % (self.X.shape[0], start, stop))
        if len(self.y.shape) > 1:
            self.y = self.y[start:stop, :]
        else:
            self.y = self.y[start:stop]
        assert self.y.shape[0] == stop - start
        
class IFMonthlyAction(dense_design_matrix.DenseDesignMatrix):
    X = None
    y = None
    def __init__(self, direction, which_set=None, start=None, stop=None, examples_per_day=32402):
        self.examples_per_day = examples_per_day
        self.cutoff = .001
        
        if IFMonthlyAction.X is None:
            Xs = []
            Rs = []
            ys = []
            gs = []
            for i in range(4):
                with open('/home/xd/data/trading/IF1503_T10_action_fullexplore_Xyg.npy.part%i' % (i,), 'rb') as f: 
                    X = np.load(f) 
                    R = np.load(f)
                    y = np.load(f)
                    g = np.load(f)
#                    if y.size > 1:
                    if True:
                        Xs.append(X)
                        Rs.append(R)        
                        ys.append(y)        
                        gs.append(g)
            X = np.vstack(Xs)
            R = np.vstack(Rs)
            y = np.concatenate(ys)
            g = np.concatenate(gs)
            print 'X.shape, R.shape, y.shape =', X.shape, R.shape, y.shape
            
#            with open('/home/xd/data/trading/IF1503_T10_action_conf-5_Xy.npy', 'rb') as f: 
#                X = np.load(f)
#                y = np.load(f)
#            print 'X.shape, y.shape =', X.shape, y.shape
            
#            for i in range(X.shape[1]):
#                remove_extreme(X[:,i], self.cutoff)
                
            assert direction in ['buy', 'sell']
            imbalance = X[:,-1]
            assert np.all(imbalance != 0)
            if direction == 'buy':
                X = X[imbalance < 0][:,:-1]
                y = y[imbalance < 0]
            elif direction == 'sell':
                X = X[imbalance > 0][:,:-1]
                y = y[imbalance > 0]
            print 'After separation: X.shape, y.shape =', X.shape, y.shape
                
            X_mean = X.mean(axis=0)
            X = X - X_mean
            X_stdev = np.sqrt(X.var(axis=0))
            X /= X_stdev
#            with open('/home/xd/data/trading/IF1503_T10_action_%s_X_mean.pkl' % (direction,), 'wb') as f:
#                cPickle.dump(X_mean, f)
#            with open('/home/xd/data/trading/IF1503_T10_action_%s_X_stdev.pkl' % (direction,), 'wb') as f:
#                cPickle.dump(X_stdev, f)
                
#            for i in range(0, X.shape[1]):
#                print X[:, i].max(), X[:, i].min(), np.abs(X[:, i]).mean()
                
#            X[:,-2] = 0  # deal_vol + deal_vol_delta: remove this unreasonable feature
            
            IFMonthlyAction.X = X.astype('float32') 
            IFMonthlyAction.y = y.reshape((y.shape[0], 1))

        super(IFMonthlyAction, self).__init__(X=IFMonthlyAction.X, y=IFMonthlyAction.y, y_labels=3)
        
        if which_set is not None:
            if which_set == 'train':
                start = 0
                stop = self.X.shape[0]*2/3
            else:
                assert which_set == 'test'
                start = self.X.shape[0]*2/3
                stop = self.X.shape[0]
        else:
            assert start is not None
            assert stop is not None

        self.X = self.X[start:stop, :]
        if len(self.y.shape) > 1:
            self.y = self.y[start:stop, :]
        else:
            self.y = self.y[start:stop]
        print 'Final: X.shape, y.shape =', self.X.shape, self.y.shape
        
class IFMonthlyShort(dense_design_matrix.DenseDesignMatrix):
    X = None
    y = None
    def __init__(self, target_type='per_step_gain', gain_range=None, 
                 which_set=None, start=None, stop=None, examples_per_day=32402):
        cutoff = .001
        if IFMonthlyShort.X is None:
            with open('/home/xd/data/trading/IF1503_cut20_raw52+ind9x3_X.npy', 'rb') as f:
                X = np.load(f)
            with open('/home/xd/data/trading/IF1503_cut20_Y20x2.npy', 'rb') as f:
                Y = np.load(f)
#            with open('/home/xd/data/trading/IF1503_T10_raw48+ind27_y20.npy', 'rb') as f:
#                X2 = np.load(f)
#                Y2 = np.load(f)
                
            print 'X.shape, Y.shape =', X.shape, Y.shape
            
#            for i in range(Y.shape[1]):
#                remove_extreme(Y[:,i], cutoff)

#            mask = np.array([1]*120 + [0]*30).astype('bool')
#            X = X[:, mask]
#            X *= mask
            
            assert target_type in ['ASV', 'RSV', 'mean_gain', 'per_step_gain', 'buy_gain', 'sell_gain']
            assert target_type == 'per_step_gain'
            if target_type == 'ASV':
#                y = Y[:, gain_range[0]:gain_range[1]].mean(axis=2).max(axis=1, keepdims=True) + \
#                    Y[:, gain_range[0]:gain_range[1]].mean(axis=2).min(axis=1, keepdims=True)
                    
                y = Y[:, gain_range[0]:gain_range[1]].max(axis=1, keepdims=True) + \
                    Y[:, gain_range[0]:gain_range[1]].min(axis=1, keepdims=True)
            if target_type == 'RSV':
#                y = Y[:, gain_range[0]:gain_range[1]].mean(axis=2).max(axis=1, keepdims=True) + \
#                    Y[:, gain_range[0]:gain_range[1]].mean(axis=2).min(axis=1, keepdims=True)
                    
                dmax = Y[:, gain_range[0]:gain_range[1]].max(axis=1, keepdims=True)
                dmin = Y[:, gain_range[0]:gain_range[1]].min(axis=1, keepdims=True)
                y = (dmax + dmin) / (dmax - dmin + .000001) 
                
            elif target_type == 'mean_gain':
                assert gain_range is not None
                y = Y[:, gain_range[0]:gain_range[1]].mean(axis=2).mean(axis=1, keepdims=True)
            elif target_type == 'per_step_gain':
                y = Y[:, gain_range[0]:gain_range[1], :]
                y = y.reshape((y.shape[0], y.shape[1]*y.shape[2]))
            elif target_type == 'buy_gain':
                y = -Y[:, gain_range[0]:gain_range[1], 0].min(axis=1, keepdims=True)
            elif target_type == 'sell_gain':
                y = Y[:, gain_range[0]:gain_range[1], 1].max(axis=1, keepdims=True)
                
            for i in range(y.shape[1]):
                remove_extreme(y[:,i], cutoff)
                
            IFMonthlyShort.X = X.astype('float32')
            IFMonthlyShort.y = y.astype('float32')

        super(IFMonthlyShort, self).__init__(X=IFMonthlyShort.X, y=IFMonthlyShort.y)
        
        if which_set is not None:
            if which_set == 'train':
                start = 0
                stop = self.X.shape[0]*2/3
            else:
                assert which_set == 'test'
                start = self.X.shape[0]*2/3
                stop = self.X.shape[0]
        else:
            assert start is not None
            assert stop is not None

        self.X = self.X[start:stop, :]
        if len(self.y.shape) > 1:
            self.y = self.y[start:stop, :]
        else:
            self.y = self.y[start:stop]
        print 'Final: X.shape, y.shape =', self.X.shape, self.y.shape
        
class IFMonthlyLong(dense_design_matrix.DenseDesignMatrix):
    X = None
    y = None
    def __init__(self, X_filter='all', target_type='ASV', gain_range=None, 
                 which_set=None, start=None, stop=None, examples_per_day=32402):
        cutoff = .001
        if IFMonthlyLong.X is None:
#            with open('/home/xd/data/trading/IF1503_cut20_raw52+ind9x3_X.npy', 'rb') as f:
            with open('/home/xd/data/trading/IF1503_cut20_43x3_X.npy', 'rb') as f:
                X = np.load(f)
            with open('/home/xd/data/trading/IF1503_cut20_Y20.npy', 'rb') as f:
                Y = np.load(f)
#            with open('/home/xd/data/trading/IF1503_T10_raw48+ind27_y20.npy', 'rb') as f:
#                X2 = np.load(f)
#                Y2 = np.load(f)
                
            print 'X.shape, Y.shape =', X.shape, Y.shape
            
            Y = Y.astype('int32')  # this line is very important, I don't know why.
            
            for i in range(Y.shape[1]):
                remove_extreme(Y[:,i], cutoff)
            
#            X = form_history(X, hist_len, hist_step)
#            print 'History formed: X.shape, Y.shape =', X.shape, Y.shape
            
            assert target_type in ['ASV', 'RSV', 'mean_gain', 'per_step_gain', 'buy_gain', 'sell_gain']
            if target_type == 'ASV':
#                y = Y[:, gain_range[0]:gain_range[1]].mean(axis=2).max(axis=1, keepdims=True) + \
#                    Y[:, gain_range[0]:gain_range[1]].mean(axis=2).min(axis=1, keepdims=True)
                    
                y = Y[:, gain_range[0]:gain_range[1]].max(axis=1, keepdims=True) + \
                    Y[:, gain_range[0]:gain_range[1]].min(axis=1, keepdims=True)
            if target_type == 'RSV':
#                y = Y[:, gain_range[0]:gain_range[1]].mean(axis=2).max(axis=1, keepdims=True) + \
#                    Y[:, gain_range[0]:gain_range[1]].mean(axis=2).min(axis=1, keepdims=True)
                    
                dmax = Y[:, gain_range[0]:gain_range[1]].max(axis=1, keepdims=True)
                dmin = Y[:, gain_range[0]:gain_range[1]].min(axis=1, keepdims=True)
                y = (dmax + dmin) / (dmax - dmin + .000001) 
                
            elif target_type == 'mean_gain':
                assert gain_range is not None
                y = Y[:, gain_range[0]:gain_range[1]].mean(axis=2).mean(axis=1, keepdims=True)
            elif target_type == 'per_step_gain':
                y = Y[:, gain_range[0]:gain_range[1], :]
                y = y.reshape((y.shape[0], y.shape[1]*y.shape[2]))
            elif target_type == 'buy_gain':
                y = -Y[:, gain_range[0]:gain_range[1], 0].min(axis=1, keepdims=True)
            elif target_type == 'sell_gain':
                y = Y[:, gain_range[0]:gain_range[1], 1].max(axis=1, keepdims=True)
                
#            for i in range(y.shape[1]):
#                remove_extreme(y[:,i], cutoff)
                
            IFMonthlyLong.X = X.astype('float32')
            IFMonthlyLong.y = y.astype('float32')

        super(IFMonthlyLong, self).__init__(X=IFMonthlyLong.X, y=IFMonthlyLong.y)
        
        if which_set is not None:
            if which_set == 'train':
                start = 0
                stop = self.X.shape[0]*2/3
            else:
                assert which_set == 'test'
                start = self.X.shape[0]*2/3
                stop = self.X.shape[0]
        else:
            assert start is not None
            assert stop is not None

        self.X = self.X[start:stop, :]
        if len(self.y.shape) > 1:
            self.y = self.y[start:stop, :]
        else:
            self.y = self.y[start:stop]
        print 'Final: X.shape, y.shape =', self.X.shape, self.y.shape
        
from preprocessor2 import long_ts_dict, short_ts_dict

class IFMonthly2(dense_design_matrix.DenseDesignMatrix):
    X = None
    y = None
    def __init__(self, long_ts=None, short_ts=None, use_long=True, use_short=True, target_type='ASV', gain_range=None, 
                 which_set=None, hist_len=1, start=None, stop=None, examples_per_day=32402):
        self_class = IFMonthly2
        if self_class.X is None:
            with open('/home/xd/data/trading/IF1503_cut20_raw+order+long+short_X.npy', 'rb') as f:
                X_raw = np.load(f)
                X_order = np.load(f)
                X_long = np.load(f)
                X_short = np.load(f)
                
            if long_ts is not None:
                long_ts_indices = [long_ts_dict[t] for t in long_ts]
                X_long = X_long[:,long_ts_indices,:]
            X_long = X_long.reshape((X_long.shape[0], -1))
            
            if short_ts is not None:
                short_ts_indices = [short_ts_dict[t] for t in short_ts]
                X_short = X_short[:,short_ts_indices,:]
            X_short = X_short.reshape((X_short.shape[0], -1))
            
            X = np.hstack([X_raw, X_order])
            if use_long:
                X = np.hstack([X, X_long])
            if use_short:
                X = np.hstack([X, X_short])
                
            with open('/home/xd/data/trading/IF1503_cut20_Y20.npy', 'rb') as f:
                Y = np.load(f)
            print 'X.shape, Y.shape =', X.shape, Y.shape
            
            Y = Y.astype('int32')  # this line is very important, I don't know why.
            for i in range(Y.shape[1]):
                remove_extreme(Y[:,i], .001)
            
            assert target_type in ['ASV',] # 'RSV', 'mean_gain', 'per_step_gain', 'buy_gain', 'sell_gain']
            if target_type == 'ASV':
                y = Y[:, gain_range[0]:gain_range[1]].max(axis=1, keepdims=True) + \
                    Y[:, gain_range[0]:gain_range[1]].min(axis=1, keepdims=True)
                
            X = form_history(X, hist_len)
            self_class.X = X.astype('float32')
            self_class.y = y.astype('float32')

        super(self_class, self).__init__(X=self_class.X, y=self_class.y)
        
        if which_set is not None:
            if which_set == 'train':
                start = 0
                stop = self.X.shape[0]*2/3
            else:
                assert which_set == 'test'
                start = self.X.shape[0]*2/3
                stop = self.X.shape[0]
        else:
            assert start is not None
            assert stop is not None

        self.X = self.X[start:stop, :]
        if len(self.y.shape) > 1:
            self.y = self.y[start:stop, :]
        else:
            self.y = self.y[start:stop]
        print 'Final: X.shape, y.shape =', self.X.shape, self.y.shape