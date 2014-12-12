import sys
import math
from pylearn2.utils import serial
from pylearn2.gui import patch_viewer

model_path = sys.argv[1]
height = int(sys.argv[2])
width = int(sys.argv[3])
model = serial.load(model_path)
W, = model.layers[0].transformer.get_params()
W = W.get_value()
#nframes = height * width / W.shape[1]
nframes = 3
kernel_shape = (int(math.sqrt(W.shape[0]/nframes)), int(math.sqrt(W.shape[0]/nframes)))
assert kernel_shape[0] * kernel_shape[1] * nframes == W.shape[0]
assert nframes * W.shape[1] == height * width

W = W.reshape((nframes, kernel_shape[0], kernel_shape[1], W.shape[1]))

pv = patch_viewer.PatchViewer(grid_shape=(height,width), patch_shape=[W.shape[1], W.shape[2]])
for i in range(W.shape[3]):
    for j in range(W.shape[0]):
        pv.add_patch(W[j,:,:,i])
pv.show()
