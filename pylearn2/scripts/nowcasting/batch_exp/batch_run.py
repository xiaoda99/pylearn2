from pylearn2.config import yaml_parse
from pylearn2.utils import serial
from ordereddict import OrderedDict

from pylearn2.datasets.cloudflow import CLOUDFLOW, get_model_base

#def get_model_base(base, hyperparams):
#    model_base = base
#    for key in hyperparams:
#        model_base += ('_' + key + str(hyperparams[key]))
#    return model_base

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
#                    OrderedDict([('rit', .2), ('sr', 1.)]),
                    OrderedDict([('rit', .3), ('sr', 1.)]),
                    OrderedDict([('rit', .4), ('sr', 1.)]),
                    OrderedDict([('rit', .5), ('sr', 1.)]),
                    OrderedDict([('rit', .6), ('sr', 1.)]),
#                    OrderedDict([('rit', .1), ('sr', 1.)]),
#                    OrderedDict([('rit', .08), ('sr', 1.)]),
#                    OrderedDict([('rit', .06), ('sr', 1.)]),
#                    OrderedDict([('rit', .04), ('sr', 1.)]),
#                    OrderedDict([('rit', 1.), ('sr', 1.)]),
#                    OrderedDict([('rit', 1.), ('sr', .8)]),
#                    OrderedDict([('rit', 1.), ('sr', .6)]),
#                    OrderedDict([('rit', 1.), ('sr', .4)]),
                    ]

base = 'longer'
hyperparams_list = [
                    OrderedDict([('interval', 2), ('sr', 1.)]),
                    OrderedDict([('interval', 4), ('sr', 1.)]),
                    OrderedDict([('interval', 7), ('sr', 1.)]),
                    ]

base = 'threshold'
hyperparams_list = [
                    OrderedDict([('th', 2), ('sr', 1.)]),
                    OrderedDict([('th', 3), ('sr', 1.)]),
                    OrderedDict([('th', 4), ('sr', 1.)]),
                    ]
base = 'low_intensity'
hyperparams_list = [
#                    OrderedDict([('ceiling', 4), ('sr', .6)]),
#                    OrderedDict([('ceiling', 3), ('sr', .6)]),
                    OrderedDict([('ceiling', 15), ('mean_ceiling', 3.), ('sr', .6)]),
                    ]

base = 'low_intensity2'
hyperparams_list = [
#                    OrderedDict([('max_int', 2.5), ('sr', .6)]),
#                    OrderedDict([('max_int', 3.0), ('sr', .6)]),
#                    OrderedDict([('max_int', 3.5), ('sr', .6)]),
#                    OrderedDict([('max_int', 4.0), ('sr', .6)]),
#                    OrderedDict([('max_int', 2.0), ('sr', .6)]),
                    OrderedDict([('max_int', 5.0), ('sr', .6)]),
                    OrderedDict([('max_int', 6.0), ('sr', .6)]),
                    ]

base = 'int_experts'
hyperparams_list = [
    OrderedDict([('trainlow', 0.), ('trainhigh', 15.), ('testlow', 0.), ('testhigh', 15.), ('max_int', 15.), ('sr', .5)]),
    OrderedDict([('trainlow', 0.), ('trainhigh', 15.), ('testlow', 0.), ('testhigh', 15.), ('max_int', 15.), ('sr', .4)]),
#    OrderedDict([('trainlow', 0.), ('trainhigh', 15.), ('testlow', 0.), ('testhigh', 15.), ('max_int', 6.), ('sr', .6)]),
#    OrderedDict([('trainlow', 0.), ('trainhigh', 15.), ('testlow', 0.), ('testhigh', 15.), ('max_int', 15.), ('sr', .6)]),
#    OrderedDict([('trainlow', 0.), ('trainhigh', 15.), ('testlow', 0.), ('testhigh', 3.), ('max_int', 15.), ('sr', .6)]),
#    OrderedDict([('trainlow', 0.), ('trainhigh', 15.), ('testlow', 3.), ('testhigh', 15.), ('max_int', 15.), ('sr', .6)]),
    
#    OrderedDict([('trainlow', 0.), ('trainhigh', 15.), ('testlow', 0.), ('testhigh', 3.), ('max_int', 2.0), ('sr', .6)]),
#    OrderedDict([('trainlow', 0.), ('trainhigh', 15.), ('testlow', 0.), ('testhigh', 3.), ('max_int', 2.5), ('sr', .6)]),
#    OrderedDict([('trainlow', 0.), ('trainhigh', 2.5), ('testlow', 0.), ('testhigh', 3.), ('max_int', 15.), ('sr', .6)]),
#    OrderedDict([('trainlow', 0.), ('trainhigh', 3.0), ('testlow', 0.), ('testhigh', 3.), ('max_int', 15.), ('sr', .6)]),
#    OrderedDict([('trainlow', 0.), ('trainhigh', 3.5), ('testlow', 0.), ('testhigh', 3.), ('max_int', 15.), ('sr', .6)]),
#    OrderedDict([('trainlow', 0.), ('trainhigh', 2.5), ('testlow', 0.), ('testhigh', 3.), ('max_int', 2.5), ('sr', .6)]),
#    OrderedDict([('trainlow', 0.), ('trainhigh', 3.0), ('testlow', 0.), ('testhigh', 3.), ('max_int', 2.5), ('sr', .6)]),
#    OrderedDict([('trainlow', 0.), ('trainhigh', 3.5), ('testlow', 0.), ('testhigh', 3.), ('max_int', 2.5), ('sr', .6)]),
#    
#    OrderedDict([('trainlow', 0.0), ('trainhigh', 15.), ('testlow', 3.), ('testhigh', 15.), ('max_int', 6.), ('sr', .6)]),
#    OrderedDict([('trainlow', 0.0), ('trainhigh', 15.), ('testlow', 3.), ('testhigh', 15.), ('max_int', 7.), ('sr', .6)]),
#    OrderedDict([('trainlow', 3.0), ('trainhigh', 15.), ('testlow', 3.), ('testhigh', 15.), ('max_int', 15.), ('sr', .6)]),
#    OrderedDict([('trainlow', 2.5), ('trainhigh', 15.), ('testlow', 3.), ('testhigh', 15.), ('max_int', 15.), ('sr', .6)]),
#    OrderedDict([('trainlow', 2.0), ('trainhigh', 15.), ('testlow', 3.), ('testhigh', 15.), ('max_int', 15.), ('sr', .6)]),
#    OrderedDict([('trainlow', 3.0), ('trainhigh', 15.), ('testlow', 3.), ('testhigh', 15.), ('max_int', 7.), ('sr', .6)]),
#    OrderedDict([('trainlow', 2.5), ('trainhigh', 15.), ('testlow', 3.), ('testhigh', 15.), ('max_int', 7.), ('sr', .6)]),
#    OrderedDict([('trainlow', 2.0), ('trainhigh', 15.), ('testlow', 3.), ('testhigh', 15.), ('max_int', 7.), ('sr', .6)]),
                    ]

base = 'sample2'
hyperparams_list = [
                    OrderedDict([('adaptive', 1), ('prob', 1.)]),
                    OrderedDict([('adaptive', 1), ('prob', .8)]),
                    OrderedDict([('adaptive', 1), ('prob', .6)]),
                    OrderedDict([('adaptive', 1), ('prob', .4)]),
                    OrderedDict([('adaptive', 1), ('prob', .2)]),
                    ]


base = 'moe'
hyperparams_list = [
    ('gpu0', OrderedDict([('testlow', 0.), ('testhigh', 2.5), ('trainlow', 0.), ('trainhigh', 2.5), ('maxint', 15.)])),
#    ('gpu0', OrderedDict([('testlow', 0.), ('testhigh', 2.5), ('trainlow', 0.), ('trainhigh', 15.), ('maxint', 15.)])),
#    ('gpu0', OrderedDict([('testlow', 0.), ('testhigh', 2.5), ('trainlow', 0.), ('trainhigh', 15.), ('maxint', 2.5)])),
#    ('gpu0', OrderedDict([('testlow', 0.), ('testhigh', 2.5), ('trainlow', 0.), ('trainhigh', 15.), ('maxint', 3.)])),
#    ('gpu0', OrderedDict([('testlow', 0.), ('testhigh', 2.5), ('trainlow', 0.), ('trainhigh', 15.), ('maxint', 4.)])),
#    ('gpu0', OrderedDict([('testlow', 0.), ('testhigh', 3.), ('trainlow', 0.), ('trainhigh', 15.), ('maxint', 15.)])),
#    ('gpu0', OrderedDict([('testlow', 0.), ('testhigh', 3.), ('trainlow', 0.), ('trainhigh', 15.), ('maxint', 3.)])),
#    ('gpu0', OrderedDict([('testlow', 0.), ('testhigh', 3.), ('trainlow', 0.), ('trainhigh', 15.), ('maxint', 4.)])),
#    ('gpu0', OrderedDict([('testlow', 0.), ('testhigh', 3.), ('trainlow', 0.), ('trainhigh', 15.), ('maxint', 5.)])),
#    ('gpu1', OrderedDict([('testlow', 0.), ('testhigh', 3.), ('trainlow', 0.), ('trainhigh', 3.), ('maxint', 15.)])),
#    ('gpu1', OrderedDict([('testlow', 0.), ('testhigh', 3.), ('trainlow', 0.), ('trainhigh', 3.), ('maxint', 3.)])),
#    ('gpu1', OrderedDict([('testlow', 0.), ('testhigh', 3.), ('trainlow', 0.), ('trainhigh', 3.), ('maxint', 4.)])),
#    ('gpu1', OrderedDict([('testlow', 0.), ('testhigh', 3.), ('trainlow', 0.), ('trainhigh', 3.), ('maxint', 5.)])),
    ]

base = 'moe_th3'
hyperparams_list = [
    ('gpu0', OrderedDict([('testlow', 0.), ('testhigh', 2.), ('trainlow', 0.), ('trainhigh', 2.), ('adapt', 0), ('sp', 0.4)])),
    ('gpu0', OrderedDict([('testlow', 0.), ('testhigh', 2.), ('trainlow', 0.), ('trainhigh', 2.), ('adapt', 0), ('sp', 0.3)])),
    ('gpu0', OrderedDict([('testlow', 0.), ('testhigh', 2.), ('trainlow', 0.), ('trainhigh', 2.), ('adapt', 0), ('sp', 0.2)])),
    ('gpu0', OrderedDict([('testlow', 0.), ('testhigh', 2.), ('trainlow', 0.), ('trainhigh', 2.), ('adapt', 0), ('sp', 0.1)])),
    ]

base = 'normalization'
hyperparams_list = [
#    ('gpu0', OrderedDict([('norm', 2), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .5)])),
#    ('gpu0', OrderedDict([('norm', 0), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .5)])),
#    ('gpu0', OrderedDict([('norm', 2), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .01)])),
#    ('gpu0', OrderedDict([('norm', 1), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .01)])),
#    ('gpu0', OrderedDict([('norm', 3), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .01)])),
#    ('gpu0', OrderedDict([('norm', 1), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .01), ('as', 1)])),
#    ('gpu0', OrderedDict([('norm', 2), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .01), ('as', 1)])),
#    ('gpu0', OrderedDict([('norm', 3), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .01), ('as', 1)])),
#    ('gpu0', OrderedDict([('norm', 1), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .01), ('as', 2)])),
#    ('gpu0', OrderedDict([('norm', 2), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .01), ('as', 2)])),
#    ('gpu0', OrderedDict([('norm', 3), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .01), ('as', 2)])),
    ('gpu0', OrderedDict([('norm', 0), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .01), ('as', 0), ('sp', .6)])),
#    ('gpu0', OrderedDict([('norm', 0), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .01), ('as', 0), ('sp', .5)])),
#    ('gpu0', OrderedDict([('norm', 0), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .01), ('as', 0), ('sp', .4)])),
#    ('gpu0', OrderedDict([('norm', 0), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .01), ('as', 0), ('sp', .3)])),
    ('gpu0', OrderedDict([('norm', 2), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .01), ('as', 0), ('sp', .6)])),
#    ('gpu0', OrderedDict([('norm', 2), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .01), ('as', 0), ('sp', .5)])),
#    ('gpu0', OrderedDict([('norm', 2), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .01), ('as', 0), ('sp', .4)])),
#    ('gpu0', OrderedDict([('norm', 2), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .01), ('as', 0), ('sp', .3)])),
    ('gpu0', OrderedDict([('norm', 3), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .01), ('as', 0), ('sp', .6)])),
#    ('gpu0', OrderedDict([('norm', 3), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .01), ('as', 0), ('sp', .5)])),
#    ('gpu0', OrderedDict([('norm', 3), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .01), ('as', 0), ('sp', .4)])),
#    ('gpu0', OrderedDict([('norm', 3), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .01), ('as', 0), ('sp', .3)])),
#    ('gpu0', OrderedDict([('norm', 2), ('minint', 0.), ('h0irange', .0002), ('h0mcn', .3), ('lr', .001), ('as', 0), ('sp', .6)])),
    ]



base = 'harder2'
hyperparams_list = [
    ('gpu0', OrderedDict([('norm', 0), ('lr', .1)])),
    ('gpu0', OrderedDict([('norm', 2), ('lr', .1)])),
]

base = 'cnn'
hyperparams_list = [
#    ('gpu0', OrderedDict([('norm', 0), ('fltr', 0), ('h0ks', 7), ('h0pad', 3), ('h1ks', 3), ('h1pad', 1), 
#                          ('h2nu', 40), ('h2mcn', 10.), ('ymcn', 10.), ('lr', .01), ('hl', 50)])),
#    ('gpu0', OrderedDict([('norm', 2), ('sp', .6), ('fltr', 0), ('h0ks', 7), ('h0pad', 3), ('h1ks', 3), ('h1pad', 1), 
#                          ('h2nu', 40), ('h2mcn', 10.), ('ymcn', 10.), ('lr', .01), ('hl', 50)])),
#    ('gpu0', OrderedDict([('norm', 0), ('sp', 1.), ('fltr', 1), ('h0ks', 7), ('h0pad', 3), ('h1ks', 3), ('h1pad', 1), 
#                          ('h2nu', 40), ('h2mcn', 10.), ('ymcn', 10.), ('lr', .1), ('hl', 50)])),
#    ('gpu0', OrderedDict([('norm', 2), ('sp', 1.), ('fltr', 1), ('h0ks', 7), ('h0pad', 3), ('h1ks', 3), ('h1pad', 1), 
#                          ('h2nu', 40), ('h2mcn', 10.), ('ymcn', 10.), ('lr', .1), ('hl', 50)])),
    ('gpu0', OrderedDict([('norm', 0), ('sp', .6), ('fltr', 0), ('h0ks', 5), ('h0pad', 2), ('h1ks', 3), ('h1pad', 1), 
                          ('h2nu', 40), ('h2mcn', .35), ('ymcn', 1.2), ('wd', 0.), ('lr', .01), ('hl', 50)])),
    ('gpu0', OrderedDict([('norm', 0), ('sp', .6), ('fltr', 0), ('h0ks', 7), ('h0pad', 3), ('h1ks', 3), ('h1pad', 1), 
                          ('h2nu', 40), ('h2mcn', .35), ('ymcn', 1.2), ('wd', 0.), ('lr', .01), ('hl', 50)])),
    ('gpu0', OrderedDict([('norm', 0), ('sp', .6), ('fltr', 0), ('h0ks', 9), ('h0pad', 4), ('h1ks', 3), ('h1pad', 1), 
                          ('h2nu', 40), ('h2mcn', .35), ('ymcn', 1.2), ('wd', 0.), ('lr', .01), ('hl', 50)])),
    ('gpu1', OrderedDict([('norm', 0), ('sp', .6), ('fltr', 0), ('h0ks', 5), ('h0pad', 2), ('h1ks', 3), ('h1pad', 1), 
                          ('h2nu', 40), ('h2mcn', 10.), ('ymcn', 10.), ('wd', .0002), ('lr', .01), ('hl', 50)])),
    ('gpu1', OrderedDict([('norm', 0), ('sp', .6), ('fltr', 0), ('h0ks', 7), ('h0pad', 3), ('h1ks', 3), ('h1pad', 1), 
                          ('h2nu', 40), ('h2mcn', 10.), ('ymcn', 10.), ('wd', .0002), ('lr', .01), ('hl', 50)])),
    ('gpu1', OrderedDict([('norm', 0), ('sp', .6), ('fltr', 0), ('h0ks', 9), ('h0pad', 4), ('h1ks', 3), ('h1pad', 1), 
                          ('h2nu', 40), ('h2mcn', 10.), ('ymcn', 10.), ('wd', .0002), ('lr', .01), ('hl', 50)])),
                    
    ('gpu1', OrderedDict([('norm', 0), ('sp', .6), ('fltr', 0), ('h0ks', 5), ('h0pad', 2), ('h1ks', 3), ('h1pad', 1), 
                          ('h2nu', 40), ('h2mcn', 10.), ('ymcn', 10.), ('wd', .0002), ('lr', .01), ('hl', 50)])),          
]

base = 'harder'
hyperparams_list = [
#    ('gpu0', OrderedDict([('norm', 2)])),
#    ('gpu0', OrderedDict([('norm', 0)])),
#    ('gpu0', OrderedDict([('norm', 2), ('lr', .1)])),
    ('gpu', OrderedDict([('norm', 0), ('lr', .1)])),
]

base = 'cnn3h'
hyperparams_list = [
    ('gpu', OrderedDict([('h0ks', 5), ('h0pad', 2), ('h2nu', 40), ('h2iip', 0.5), ('h2is', 2.),
                         ('h2mcn', 10.), ('ymcn', 10.), ('wd', .0002), ('lr', .01)])),
]

base = 'cnn2h'
hyperparams_list = [
    ('gpu0', OrderedDict([('h0ks', 5), ('h0pad', 2), ('h1ps', 2), ('ymcn', 99.), ('wd', .0002), ('lr', .01)])),
    ('gpu0', OrderedDict([('h0ks', 5), ('h0pad', 2), ('h1ps', 1), ('ymcn', 99.), ('wd', .0002), ('lr', .01)])),
    ('gpu1', OrderedDict([('h0ks', 5), ('h0pad', 2), ('h1ps', 3), ('ymcn', 99.), ('wd', .0002), ('lr', .01)])),
    ('gpu1', OrderedDict([('h0ks', 5), ('h0pad', 2), ('h1ps', 6), ('ymcn', 99.), ('wd', .0002), ('lr', .01)])),
]

if __name__ == "__main__":
    import sys
    device = sys.argv[1]
    yaml_template = open(base + '_template.yaml', 'r').read()
    results_path = base + '_' + device + '_results.txt'
    for (dev, hyperparams) in hyperparams_list:
        if dev == device:
            model_base = get_model_base(base, hyperparams)
            hyperparams.update({'save_base' : model_base})
            yaml = yaml_template % (hyperparams)
            with open(model_base + '.yaml', 'w') as f:
                f.write(yaml)
    #        print yaml
            train = yaml_parse.load(yaml)
            train.main_loop()
        
            model_path = model_base + '_best.pkl'
            
            print_results(model_path, results_path)