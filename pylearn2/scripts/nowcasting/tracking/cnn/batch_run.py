from pylearn2.config import yaml_parse
    
mlp_yaml_template = open('mlp_36-48m_3x24x24-200_template.yaml', 'r').read()
hyper_params_mlp = [
#                    {'base' : 'mlp', 'track' : 1, 'sr00' : 1.},
#                    {'base' : 'mlp', 'track' : 1, 'sr00' : .8},
#                    {'base' : 'mlp', 'track' : 1, 'sr00' : .6},
#                    {'base' : 'mlp', 'track' : 1, 'sr00' : .4},
#                    {'base' : 'mlp_rel_sample', 'track' : 1, 'sr00' : 1.},
#                    {'base' : 'mlp_rel_sample', 'track' : 1, 'sr00' : .8},
#                    {'base' : 'mlp_rel_sample', 'track' : 1, 'sr00' : .6},
#                    {'base' : 'mlp_rel_sample', 'track' : 1, 'sr00' : .4},
                    {'base' : 'mlp', 'track' : 0, 'sr00' : 1.},
                    {'base' : 'mlp', 'track' : 0, 'sr00' : .8},
                    {'base' : 'mlp', 'track' : 0, 'sr00' : .6},
                    {'base' : 'mlp', 'track' : 0, 'sr00' : .4},
                    ]

for hyper_params in hyper_params_mlp:
    mlp_yaml = mlp_yaml_template % (hyper_params)
    train = yaml_parse.load(mlp_yaml)
    train.main_loop()