#from __future__ import print_function
import numpy as np
import pylab as plt

from pylearn2.datasets.cloudflow import CLOUDFLOW
from pylearn2.utils import py_integer_types
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from theano import config
import theano.tensor as T
from theano import function
import sys

batch_size = 500
prob_threshold = 0.5
norm = 0

def build_pred_fn(model):
#        model = serial.load(model_path)
    model.set_batch_size(batch_size)
    print 'compiling function ...'
    X = model.get_input_space().make_batch_theano()
    y = model.fprop(X)
    fn = function([X], y,allow_input_downcast=True)
    print 'done.'
    return fn

test = CLOUDFLOW(  
             which_set='test',
             num_examples=700000,
             threshold=2,
             train_frame_size = (3,25,25),
             predict_frame_size = (2,1,1),
             predict_interval = 2,
             examples_per_image = 100,
             intensity_range= [0., 2.],
             max_intensity= 15.,
             normalization=norm,
             adaptive_sampling= 0,
             sample_prob= 1.,
#             filter=True,
             show_mode=False
             )

#from batch_run import base, hyperparams_list
#test.show_random_examples(base, hyperparams_list)
        
def test_range(int_range, pred_fn, prob_threshold):
    test = CLOUDFLOW(  
                 which_set='train',
                 num_examples=700000,
                 threshold=2,
                 pool_xy=2,
                 pool_t=1,
                 train_frame_size = (3,25,25),
                 cropped_size= (3, 12, 12),
                 predict_frame_size = (2,1,1),
                 predict_interval = 2,
                 examples_per_image = 100,
                 intensity_range= int_range,
                 max_intensity= 15.,
                 normalization=norm,
                 adaptive_sampling= 0,
                 sample_prob= .6,
#                 filter_model='norm2_200-100_mom0.9_lr0.01_best.pkl',
#                 filter_prob_range=[0.2, 0.8],
                 )
    preds = []
#    rains = []
    pred_and_rains =[]
    assert isinstance(test.X.shape[0], py_integer_types)
#    assert isinstance(batch_size, py_integer_types)
    iterator = test.iterator(mode = 'even_sequential',
                            batch_size = batch_size,
                            data_specs = model.cost_from_X_data_specs())
    X = model.get_input_space().make_batch_theano()
    for item in iterator:
        x_arg, y_arg = item
        if X.ndim > 2:
            x_arg = test.get_topological_view(x_arg)
        res = pred_fn(x_arg, y_arg)
        preds.append(res[0])
#        rains.append(res[1])
        pred_and_rains.append(res[1])
    npred = sum(preds)
#    nrain = sum(rains)
    nrain = test.y.sum() 
    npred_and_rain = sum(pred_and_rains)
#    precision = npred_and_rain / npred
#    recall = npred_and_rain / nrain
#    f1 = 2. * precision * recall / (precision + recall)
#    print npred, nrain, npred_and_rain, 1. - precision, 1. - recall, f1
    
    nexamples = test.X.shape[0]
    npred_flow = test.y1.sum()
    
    nfalse_pos = npred - npred_and_rain
    nfalse_neg = nrain - npred_and_rain
    npred_and_rain_flow = np.sum(test.y * test.y1)
    nfalse_pos_flow = npred_flow - npred_and_rain_flow
    nfalse_neg_flow = nrain - npred_and_rain_flow
    return nexamples, nrain, npred_flow, npred, npred_and_rain, \
        nfalse_pos_flow, nfalse_neg_flow, nfalse_pos, nfalse_neg

def test_intensity(model, prob_threshold):
    model.set_batch_size(batch_size)
    
    X = model.get_input_space().make_batch_theano()
    y = model.get_output_space().make_batch_theano()
    yout = model.fprop(X)
    
    prediction = T.gt(yout, prob_threshold)
    pred = T.cast(prediction, config.floatX).sum()  # mean() and sum() are the same
#    rain = T.cast(y, config.floatX).sum()
    pred_and_rain = T.cast(T.eq(prediction + y, 2), config.floatX).sum() 
    
    pred_fn = function([X,y],[pred, pred_and_rain])
    
    nranges = 5
    nexamples = np.zeros(nranges)
    nrain = np.zeros(nranges)
    npred_flow = np.zeros(nranges)
    npred = np.zeros(nranges)
    npred_and_rain = np.zeros(nranges)
    
    nfalse_pos_flow = np.zeros(nranges)
    nfalse_neg_flow = np.zeros(nranges)
    nfalse_pos = np.zeros(nranges)
    nfalse_neg = np.zeros(nranges)

    for i in range(nranges):
        int_range = [float(i), float(i+1)]
        nexamples[i], nrain[i], npred_flow[i], npred[i], npred_and_rain[i],\
            nfalse_pos_flow[i], nfalse_neg_flow[i], nfalse_pos[i], nfalse_neg[i] = \
            test_range(int_range, pred_fn, prob_threshold)
        print int_range
        print nexamples[i], nrain[i], npred_flow[i], npred[i], npred_and_rain[i]
    
#    fig, ax = plt.subplots()
    plt.subplot(2, 1, 1)
    index = np.arange(nranges)
    bar_width = 0.15
    rects1 = plt.bar(index, nexamples, bar_width, color='r', label='nexamples')
    rects2 = plt.bar(index+bar_width, nrain, bar_width, color='y', label='nrain')
    rects3 = plt.bar(index+2*bar_width, npred_flow, bar_width, color='g', label='npred_flow')
    rects4 = plt.bar(index+3*bar_width, npred, bar_width, color='b', label='npred')
    plt.legend()
    plt.tight_layout()
#    plt.show()
    
#    fig, ax = plt.subplots()
    plt.subplot(2, 1, 2)
    index = np.arange(nranges)
    bar_width = 0.15
    rects1 = plt.bar(index, nfalse_pos_flow, bar_width, color='r', label='nfalse_pos_flow')
    rects2 = plt.bar(index+bar_width, nfalse_neg_flow, bar_width, color='g', label='nfalse_neg_flow')
    rects3 = plt.bar(index+2*bar_width, nfalse_pos, bar_width, color='y', label='nfalse_pos')
    rects4 = plt.bar(index+3*bar_width, nfalse_neg, bar_width, color='b', label='nfalse_neg')
    plt.legend()
    plt.tight_layout()
    plt.show()

def test_prob(test, model):
    pred_fn = build_pred_fn(model)
    
    iterator = test.iterator(mode = 'even_sequential',
                            batch_size = batch_size,
                            data_specs = model.cost_from_X_data_specs())
    probs = []
    for item in iterator:
        x_arg, y_arg = item
#        if X.ndim > 2:
#            x_arg = test.get_topological_view(x_arg)
        probs += list(pred_fn(x_arg)[:,0])
    
    probs = np.array(probs)    
    y = test.y[:,0]
    print 'probs.shape, y.shape =', probs.shape, y.shape
    min_len = min(probs.shape[0], y.shape[0])
    probs = probs[:min_len]
    y = y[:min_len]
    assert probs.shape == y.shape
    
    n_prob_ranges = 10
    n_right = np.zeros(n_prob_ranges, dtype='int32')
    n_wrong = np.zeros(n_prob_ranges, dtype='int32')
    total = 0
    for i in range(n_prob_ranges):
        prob_range = [i * 0.1, (i + 1) * 0.1]
        in_range = (probs >= prob_range[0]) * (probs < prob_range[1])
        right = (y == 1); wrong = (y == 0)
        if prob_range[1] <= 0.5:
            right, wrong = wrong, right
        n_right[i] = (in_range * right).sum()
        n_wrong[i] = (in_range * wrong).sum()
        total += n_right[i]
        total += n_wrong[i]
    assert total == probs.shape[0]
    print 'n_wrong.sum(), total =', n_wrong.sum(), total
        
    ind = np.arange(n_prob_ranges)    # the x locations for the groups
    width = 0.35       # the width of the bars: can also be len(x) sequence
    
    p1 = plt.bar(ind, n_wrong, width, color='r', label='n_wrong')
    p2 = plt.bar(ind, n_right, width, color='g', bottom=n_wrong, label='n_right')
    
#    plt.ylabel('Scores')
#    plt.title('Scores by group and gender')
#    plt.xticks(ind+width/2., ('G1', 'G2', 'G3', 'G4', 'G5') )
#    plt.yticks(np.arange(0,81,10))
#    plt.legend( (p1[0], p2[0]), ('Men', 'Women') )
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def build_stat_fn(model, prob_threshold=0.5):
    model.set_batch_size(batch_size)
    
    X = model.get_input_space().make_batch_theano()
    y = model.get_output_space().make_batch_theano()
    yout = model.fprop(X)
    
    prediction = T.gt(yout, prob_threshold)
    pred = T.cast(prediction, config.floatX).sum()  # mean() and sum() are the same
    rain = T.cast(y, config.floatX).sum()
    pred_and_rain = T.cast(T.eq(prediction + y, 2), config.floatX).sum() 
    
    incorrect = T.neq(y, prediction).max(axis=1)
    misclass = T.cast(incorrect, config.floatX).mean()
    nll = model.cost(y, yout)
    
    fn = function([X,y],[pred, rain, pred_and_rain, misclass, nll])
    return fn

def build_stat_fn_ensemble(models, prob_threshold=0.5):
    assert len(models) == 2
    for model in models:
        model.set_batch_size(batch_size)
    
    X = models[0].get_input_space().make_batch_theano()
    y = models[0].get_output_space().make_batch_theano()
#    youts = [model.fprop(X) for model in models]
    
    yout = (0.3 * models[0].fprop(X) + 0.7 * models[1].fprop(X)) / 1.
    prediction = T.gt(yout, prob_threshold)
    pred = T.cast(prediction, config.floatX).sum()  # mean() and sum() are the same
    rain = T.cast(y, config.floatX).sum()
    pred_and_rain = T.cast(T.eq(prediction + y, 2), config.floatX).sum() 
    
    incorrect = T.neq(y, prediction).max(axis=1)
    misclass = T.cast(incorrect, config.floatX).mean()
#    nll = model.cost(y, yout)
    
    fn = function([X,y],[pred, rain, pred_and_rain, misclass])
    return fn

def test_accuracy_ensemble(test, models, prob_threshold):
    stat_fn = build_stat_fn_ensemble(models, prob_threshold)
    
    n_pred = 0.; n_rain = 0.; n_pred_and_rain = 0.; misclass = 0.; nll = 0.
    n_batches = 0
    iterator = test.iterator(mode = 'even_sequential',
                            batch_size = batch_size,
                            data_specs = model.cost_from_X_data_specs())
    for item in iterator:
        x_arg, y_arg = item
#        if X.ndim > 2:
#            x_arg = test.get_topological_view(x_arg)
        stat = stat_fn(x_arg, y_arg)
        n_pred += stat[0]
        n_rain += stat[1]
        n_pred_and_rain += stat[2]
        misclass += stat[3]
#        print stat[3]
#        nll += stat[4]
        n_batches += 1
    n_pred /= n_batches
    n_rain /= n_batches
    n_pred_and_rain /= n_batches
    misclass /= n_batches
#    nll /= n_batches
    
    n_false_pos = n_pred - n_pred_and_rain
    n_false_neg = n_rain - n_pred_and_rain
    
    precision = n_pred_and_rain / n_pred
    recall = n_pred_and_rain / n_rain
    f1 = 2*precision*recall / (precision + recall)
    
    print 'n_pred, n_pred_and_rain, n_rain =', n_pred, n_pred_and_rain, n_rain
    print 'n_false_pos, n_false_neg, misclass, nll =', n_false_pos, n_false_neg, misclass, nll
    print 'precision, recall, f1 =', precision, recall, f1

def test_accuracy(test, model, prob_threshold):
    stat_fn = build_stat_fn(model, prob_threshold)
    
    n_pred = 0.; n_rain = 0.; n_pred_and_rain = 0.; misclass = 0.; nll = 0.
    n_batches = 0
    iterator = test.iterator(mode = 'even_sequential',
                            batch_size = batch_size,
                            data_specs = model.cost_from_X_data_specs())
    X = model.get_input_space().make_batch_theano()
    for item in iterator:
        x_arg, y_arg = item
#        print ' x_arg.shape =', x_arg.shape
#        if X.ndim > 2:
#            x_arg = test.get_topological_view(x_arg)
        stat = stat_fn(x_arg, y_arg)
        n_pred += stat[0]
        n_rain += stat[1]
        n_pred_and_rain += stat[2]
        misclass += stat[3]
#        print stat[3]
        nll += stat[4]
        n_batches += 1
    n_pred /= n_batches
    n_rain /= n_batches
    n_pred_and_rain /= n_batches
    misclass /= n_batches
    nll /= n_batches
    
    n_false_pos = n_pred - n_pred_and_rain
    n_false_neg = n_rain - n_pred_and_rain
    
    precision = n_pred_and_rain / n_pred
    recall = n_pred_and_rain / n_rain
    f1 = 2*precision*recall / (precision + recall)
    
    print 'n_pred, n_pred_and_rain, n_rain =', n_pred, n_pred_and_rain, n_rain
    print 'n_false_pos, n_false_neg, misclass, nll =', n_false_pos, n_false_neg, misclass, nll
    print 'precision, recall, f1 =', precision, recall, f1
    
    n_pred_flow = test.y1.sum()
    n_rain_flow = test.y.sum()
    n_pred_and_rain_flow = (test.y1 * test.y).sum()
    
    n_false_pos_flow = n_pred_flow - n_pred_and_rain_flow
    n_false_neg_flow = n_rain_flow - n_pred_and_rain_flow
    misclass_flow = (n_false_pos_flow + n_false_neg_flow) * 1. / test.y1.shape[0]
    
    precision_flow = n_pred_and_rain_flow * 1. / n_pred_flow
    recall_flow = n_pred_and_rain_flow * 1. / n_rain_flow
    f1_flow = 2.*precision_flow*recall_flow / (precision_flow + recall_flow)
    
    print '\nFlow:'
    print 'n_pred, n_pred_and_rain, n_rain =', n_pred_flow, n_pred_and_rain_flow, n_rain_flow
    print 'n_false_pos, n_false_neg, misclass =', n_false_pos_flow, n_false_neg_flow, misclass_flow
    print 'precision, recall, f1 =', precision_flow, recall_flow, f1_flow
    
if __name__ == '__main__':
    _, model_path, norm, prob_threshold = sys.argv
    prob_threshold = float(prob_threshold)
    norm = int(norm)
    model = serial.load(model_path)
    #models = [model, serial.load(model_path2)]
    
#    test_intensity(model, prob_threshold)
#    test_prob(test, model)
    test_accuracy(test, model, prob_threshold)
#    test_accuracy_ensemble(test, models, prob_threshold)
    

