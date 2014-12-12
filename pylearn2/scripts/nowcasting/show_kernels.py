import sys
from pylearn2.utils import serial
from pylearn2.gui import patch_viewer

model_path = sys.argv[1]
height = int(sys.argv[2])
width = int(sys.argv[3])

model = serial.load(model_path)
W, = model.layers[0].transformer.get_params()
W = W.get_value()

assert height * width == W.shape[0] * W.shape[3]
pv = patch_viewer.PatchViewer(grid_shape=(height,width), patch_shape=[W.shape[1], W.shape[2]])
for i in range(W.shape[3]):
    for j in range(W.shape[0]):
        pv.add_patch(W[j, :, :, i])
pv.show()
