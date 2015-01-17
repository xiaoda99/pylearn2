from pylearn2.config import yaml_parse
from pylearn2.utils import serial
from ordereddict import OrderedDict

def get_model_base(base, hyperparams):
    model_base = base
    for key in hyperparams:
        model_base += ('_' + key + str(hyperparams[key]))
    return model_base

def print_results(model_path, results_path):
    f = open(results_path, 'a+')
    
    model = serial.load(model_path)
    monitor = model.monitor
    channels = monitor.channels
    keys = ['train_y_misclass', 'valid_y_misclass', 'test_y_misclass']
    for key in keys:
        print >>f, str(channels[key].val_record[-1]) + '\t',
        
    # added by XD
    datasets = ('test',)       
    for which_set in datasets:
        predict = channels[which_set + '_y_' + 'predict'].val_record[-1]
        rain = channels[which_set + '_y_' + 'rain'].val_record[-1]
        predict_and_rain = channels[which_set + '_y_' + 'predict_and_rain'].val_record[-1]
        precision = predict_and_rain * 1. / predict
        recall = predict_and_rain * 1. / rain
        f1 = 2. * precision * recall / (precision + recall)
        print >>f, str(1. - precision) + '\t',
        print >>f, str(1. - recall) + '\t',
        print >>f, str(f1) + '\t',
#    print >>f, '\n'
    print >>f, model_path
    f.close()
        
base = 'cnn'
#hyperparams_list = [
#                    {'base' : 'mlp', 'track' : 1, 'sr00' : 1.},
#                    {'base' : 'mlp', 'track' : 1, 'sr00' : .8},
#                    {'base' : 'mlp', 'track' : 1, 'sr00' : .6},
#                    {'base' : 'mlp', 'track' : 1, 'sr00' : .4},
#                    {'base' : 'mlp_rel_sample', 'track' : 1, 'sr00' : 1.},
#                    {'base' : 'mlp_rel_sample', 'track' : 1, 'sr00' : .8},
#                    {'base' : 'mlp_rel_sample', 'track' : 1, 'sr00' : .6},
#                    {'base' : 'mlp_rel_sample', 'track' : 1, 'sr00' : .4},
#                    {'base' : 'mlp', 'track' : 0, 'sr00' : 1.},
#                    {'base' : 'mlp', 'track' : 0, 'sr00' : .8},
#                    {'base' : 'mlp', 'track' : 0, 'sr00' : .6},
#                    {'base' : 'mlp', 'track' : 0, 'sr00' : .4},
#                    ]
hyperparams_list = [
                    OrderedDict([('track', 1), ('h1pool', 1), ('wd', 0.0001), ('sr', .6)]),
                    OrderedDict([('track', 1), ('h1pool', 2), ('wd', 0.0001), ('sr', .6)]),
                    OrderedDict([('track', 1), ('h1pool', 3), ('wd', 0.0001), ('sr', .6)]),
                    OrderedDict([('track', 1), ('h1pool', 6), ('wd', 0.0001), ('sr', .6)]),
                    ]
base = 'targets'
hyperparams_list = [
                    OrderedDict([('track', 1), ('th', 1), ('sr', 1.)]),
                    OrderedDict([('track', 1), ('th', 2), ('sr', 1.)]),
                    OrderedDict([('track', 0), ('th', 1), ('sr', 1.)]),
                    OrderedDict([('track', 0), ('th', 2), ('sr', 1.)]),
                    ]

base = 'diff'
hyperparams_list = [
                    OrderedDict([('track', 1), ('diff', 1), ('sr', 1.)]),
                    OrderedDict([('track', 0), ('diff', 1), ('sr', 1.)]),
                    ]

base = 'normalization'
hyperparams_list = [
#                    OrderedDict([('track', 1), ('norm', 1), ('h0_mcn', 100.), ('y_mcn', 100.), ('wd', 0.0001), ('sr', 1.)]),
#                    OrderedDict([('track', 1), ('norm', 1), ('h0_mcn', .4), ('y_mcn', 100.), ('wd', 0), ('sr', 1.)]),
#                    OrderedDict([('track', 1), ('norm', 1), ('h0_mcn', .2), ('y_mcn', 100.), ('wd', 0), ('sr', 1.)]),
                    OrderedDict([('track', 1), ('norm', 1), ('h0_mcn', .1), ('y_mcn', 100.), ('wd', 0), ('sr', 1.)]),
                    OrderedDict([('track', 1), ('norm', 1), ('h0_mcn', .05), ('y_mcn', 100.), ('wd', 0), ('sr', 1.)]),
                    OrderedDict([('track', 1), ('norm', 0), ('h0_mcn', .4), ('y_mcn', 100.), ('wd', 0), ('sr', 1.)]),
                    OrderedDict([('track', 1), ('norm', 0), ('h0_mcn', .2), ('y_mcn', 100.), ('wd', 0), ('sr', 1.)]),
                    OrderedDict([('track', 1), ('norm', 0), ('h0_mcn', .1), ('y_mcn', 100.), ('wd', 0), ('sr', 1.)]),
                    OrderedDict([('track', 1), ('norm', 0), ('h0_mcn', .05), ('y_mcn', 100.), ('wd', 0), ('sr', 1.)]),
                    ]

base = 'sampling'
hyperparams_list = [
                    OrderedDict([('rit', .20), ('sr', 1.)]),
                    OrderedDict([('rit', .18), ('sr', 1.)]),
                    OrderedDict([('rit', .16), ('sr', 1.)]),
                    OrderedDict([('rit', .14), ('sr', 1.)]),
                    OrderedDict([('rit', .12), ('sr', 1.)]),
#                    OrderedDict([('rit', .1), ('sr', 1.)]),
#                    OrderedDict([('rit', .08), ('sr', 1.)]),
#                    OrderedDict([('rit', .06), ('sr', 1.)]),
#                    OrderedDict([('rit', .04), ('sr', 1.)]),
#                    OrderedDict([('rit', 1.), ('sr', 1.)]),
#                    OrderedDict([('rit', 1.), ('sr', .8)]),
#                    OrderedDict([('rit', 1.), ('sr', .6)]),
#                    OrderedDict([('rit', 1.), ('sr', .4)]),
                    ]

yaml_template = open(base + '_template.yaml', 'r').read()
results_path = base + '_results.txt'
for hyperparams in hyperparams_list:
    model_base = get_model_base(base, hyperparams)
    hyperparams.update({'save_base' : model_base})
    yaml = yaml_template % (hyperparams)
    train = yaml_parse.load(yaml)
    train.main_loop()
    
    model_path = model_base + '_best.pkl'
    
    print_results(model_path, results_path)