#!/usr/bin/env python
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"
import sys
from pylearn2.utils import serial
for model_path in sys.argv[1:]:
    if len(sys.argv) > 2:
        print model_path
    model = serial.load(model_path)
    monitor = model.monitor
    channels = monitor.channels
    print 'epochs seen: ',monitor._epochs_seen
    print 'time trained: ',max(channels[key].time_record[-1] for key in channels)
    for key in sorted(channels.keys()):
        print key, ':', channels[key].val_record[-1]
        
    # added by XD
    datasets = ('train', 'valid', 'test')
    for which_set in datasets:
        if not (which_set + '_y_' + 'predict' in channels and 
                which_set + '_y_' + 'rain' in channels and 
                which_set + '_y_' + 'predict_and_rain' in channels):
            import sys
            sys.exit()
        
    for which_set in datasets:
        predict = channels[which_set + '_y_' + 'predict'].val_record[-1]
        rain = channels[which_set + '_y_' + 'rain'].val_record[-1]
        predict_and_rain = channels[which_set + '_y_' + 'predict_and_rain'].val_record[-1]
        precision = predict_and_rain * 1. / predict
        recall = predict_and_rain * 1. / rain
        f1 = 2. * precision * recall / (precision + recall)
        print which_set + '_y_' + 'false_positive', ':', 1. - precision
        print which_set + '_y_' + 'false_negative', ':', 1. - recall
        print which_set + '_y_' + 'false_f1', ':', f1
