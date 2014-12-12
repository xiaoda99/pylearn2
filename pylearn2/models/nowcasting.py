import numpy as np
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from theano.gof.op import get_debug_values

from pylearn2.models.mlp import Layer  #, Conv3DLinear
from pylearn2.space import VectorSpace, Conv2DSpace
from pylearn2.utils import sharedX
from pylearn2.linear import conv2d
from pylearn2.linear.conv2d_c01b import Conv2D

class ConvRectifiedLinearC01B(Layer):
    def __init__(self,
                 kernel_shape,
                 output_channels,
                 layer_name,
                 irange,
                 max_kernel_norm = None,
                 init_bias=0.,
                 border_mode = 'valid',
                 pad = 0,
                 partial_sum = 1,
                 kernel_stride=(1,1)):

        super(ConvRectifiedLinearC01B, self).__init__()
        self.__dict__.update(locals())
        del self.self
        
    def set_input_space(self, space):
        self.input_space = space
        rng = self.mlp.rng
        
        # Make sure number of channels is supported by cuda-convnet
        # (multiple of 4 or <= 3)
        # If not supported, pad the input with dummy channels
        ch = self.input_space.num_channels
        assert ch <= 3 or ch % 4 == 0
        
        if self.border_mode == 'valid':
            output_shape = [(self.input_space.shape[0] - self.kernel_shape[0]) / self.kernel_stride[0] + 1,
                (self.input_space.shape[1] - self.kernel_shape[1]) / self.kernel_stride[1] + 1]
        elif self.border_mode == 'full':
            output_shape = [(self.input_space.shape[0] +  self.kernel_shape[0]) / self.kernel_stride[0] - 1,
                    (self.input_space.shape[1] + self.kernel_shape[1]) / self.kernel_stride_stride[1] - 1]

        self.output_space = Conv2DSpace(shape=output_shape,
                num_channels = self.output_channels,
                axes = ('c', 0, 1, 'b'))

        rng = self.mlp.rng
        W = sharedX( rng.uniform(-self.irange,self.irange,(self.input_space.num_channels, \
               self.kernel_shape[0], self.kernel_shape[1], self.output_channels)))
        
        self.transformer = Conv2D(filters = W,
            input_axes = self.input_space.axes,
            output_axes = self.output_space.axes,
            kernel_stride = self.kernel_stride, pad=self.pad,
            message = "", partial_sum=self.partial_sum)

        W, = self.transformer.get_params()
        W.name = self.layer_name + '_W'
        
        self.b = sharedX(np.zeros((self.output_space.num_channels)) + self.init_bias)
        self.b.name = self.layer_name + '_b'

        print 'Input shape: ', self.input_space.shape
        print 'Output space: ', self.output_space.shape

    def _modify_updates(self, updates):

        if self.max_kernel_norm is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=(0,1,2)))
                desired_norms = T.clip(row_norms, 0, self.max_kernel_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + row_norms)).dimshuffle('x', 'x', 'x', 0)
                
    def get_params(self):
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval
    
    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.sqr(W).sum()
            
    def get_monitoring_channels(self):
        #return OrderedDict()  #XD debug
        W ,= self.transformer.get_params()

        assert W.ndim == 4

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=(0,1,2)))

        abs_W = T.abs_(W)   #XD add
        abs_b = T.abs_(self.b)   #XD add

        return OrderedDict([
                            ('kernel_norms_min'  , row_norms.min()),
                            ('kernel_norms_mean' , row_norms.mean()),
                            ('kernel_norms_max'  , row_norms.max()),
                            #('abs_W_mean'  , abs_W.mean()),         #XD add
                            #('abs_b_mean'  , abs_b.mean()),         #XD add
                            ])
        
    def fprop(self, state_below):
        # mlp 1st layer, standard conv
        self.input_space.validate(state_below)

        b = self.b.dimshuffle(0, 'x', 'x', 'x') # ('c',) -> ('c', 0, 1, 'b')

        z = self.transformer.lmul(state_below) + b
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        d = z * (z > 0.)
        self.output_space.validate(d)
        assert self.output_space.num_channels % 16 == 0
        return d

class ConvRectifiedLinear(Layer):
    def __init__(self,
                 output_channels,
                 kernel_shape,
                 layer_name,
                 irange = None,
                 border_mode = 'valid',
                 sparse_init = None,
                 include_prob = 1.0,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 left_slope = 0.0,
                 max_kernel_norm = None,
                 kernel_stride=(1, 1)):
        
        if (irange is None) and (sparse_init is None):
            raise AssertionError("You should specify either irange or sparse_init when calling the constructor of ConvRectifiedLinear.")
        elif (irange is not None) and (sparse_init is not None):
            raise AssertionError("You should specify either irange or sparse_init when calling the constructor of ConvRectifiedLinear and not both.")

        self.__dict__.update(locals())
        del self.self

    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! """

        self.input_space = space
        rng = self.mlp.rng

        if self.border_mode == 'valid':
            output_shape = [(self.input_space.shape[0] - self.kernel_shape[0]) / self.kernel_stride[0] + 1,
                (self.input_space.shape[1] - self.kernel_shape[1]) / self.kernel_stride[1] + 1]
        elif self.border_mode == 'full':
            output_shape = [(self.input_space.shape[0] +  self.kernel_shape[0]) / self.kernel_stride[0] - 1,
                    (self.input_space.shape[1] + self.kernel_shape[1]) / self.kernel_stride_stride[1] - 1]

        self.output_space = self.detector_space = Conv2DSpace(shape=output_shape,
                num_channels = self.output_channels,
                axes = ('b', 'c', 0, 1))

        if self.irange is not None:
            assert self.sparse_init is None
            self.transformer = conv2d.make_random_conv2D(
                    irange = self.irange,
                    input_space = self.input_space,
                    output_space = self.detector_space,
                    kernel_shape = self.kernel_shape,
                    batch_size = self.mlp.batch_size,
                    subsample = self.kernel_stride,
                    border_mode = self.border_mode,
                    rng = rng)
        elif self.sparse_init is not None:
            self.transformer = conv2d.make_sparse_random_conv2D(
                    num_nonzero = self.sparse_init,
                    input_space = self.input_space,
                    output_space = self.detector_space,
                    kernel_shape = self.kernel_shape,
                    batch_size = self.mlp.batch_size,
                    subsample = self.kernel_stride,
                    border_mode = self.border_mode,
                    rng = rng)
        W, = self.transformer.get_params()
        W.name = 'W'

        self.b = sharedX(self.detector_space.get_origin() + self.init_bias)
        self.b.name = 'b'

        print 'Input shape: ', self.input_space.shape
        print 'Output space: ', self.output_space.shape

    def censor_updates(self, updates):

        if self.max_kernel_norm is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=(1,2,3)))
                desired_norms = T.clip(row_norms, 0, self.max_kernel_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + row_norms)).dimshuffle(0, 'x', 'x', 'x')


    def get_params(self):
        assert self.b.name is not None
        W ,= self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    def get_l1_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W ,= self.transformer.get_params()
        return coeff * abs(W).sum()

    def set_weights(self, weights):
        W, = self.transformer.get_params()
        W.set_value(weights)

    def set_biases(self, biases):
        self.b.set_value(biases)

    def get_biases(self):
        return self.b.get_value()

    def get_weights_format(self):
        return ('v', 'h')

    def get_weights_topo(self):
        outp, inp, rows, cols = range(4)
        raw = self.transformer._filters.get_value()

        return np.transpose(raw, (outp,rows,cols,inp))

    def get_monitoring_channels(self):

        W ,= self.transformer.get_params()

        assert W.ndim == 4

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=(1,2,3)))

        return OrderedDict([
                            ('kernel_norms_min'  , row_norms.min()),
                            ('kernel_norms_mean' , row_norms.mean()),
                            ('kernel_norms_max'  , row_norms.max()),
                            ])

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        z = self.transformer.lmul(state_below) + self.b
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        d = z * (z > 0.) + self.left_slope * z * (z < 0.)

        self.detector_space.validate(d)
        return d
        
class PretrainedMLPConv(Layer):
    def __init__(self,
                 kernel_shape,
                 kernel_stride,
                 detector_channels,
                 expand_shape,
                 micro_mlp,
                 layer_name,
                 border_mode = 'valid'):

        self.__dict__.update(locals())
        del self.self
        
    def set_input_space(self, space):
        self.input_space = space
        rng = self.mlp.rng
        
        # Make sure number of channels is supported by cuda-convnet
        # (multiple of 4 or <= 3)
        # If not supported, pad the input with dummy channels
        #ch = self.input_space.num_channels
        #assert ch <= 3 or ch % 4 == 0
        
        params = self.micro_mlp.get_params()   
        self.b_h0 = params[1]
        #self.W_h1 = params[2]
        #self.b_h1 = params[3]
        self.W_y = params[2]
        self.b_y = params[3]
        W_h0 = params[0].get_value().reshape((self.input_space.num_channels,
                                       self.kernel_shape[0],
                                       self.kernel_shape[1],
                                       self.detector_channels))
        W_h0 = W_h0.transpose((3, 0, 1, 2))
        self.W_h0 = sharedX(W_h0, params[0].name)
        
        if self.border_mode == 'valid':
            output_shape = [(self.input_space.shape[0] - self.kernel_shape[0]) / self.kernel_stride[0] + 1,
                (self.input_space.shape[1] - self.kernel_shape[1]) / self.kernel_stride[1] + 1]
        elif self.border_mode == 'full':
            output_shape = [(self.input_space.shape[0] +  self.kernel_shape[0]) / self.kernel_stride[0] - 1,
                    (self.input_space.shape[1] + self.kernel_shape[1]) / self.kernel_stride_stride[1] - 1]

        self.detector_space = Conv2DSpace(shape=output_shape,
                num_channels = self.detector_channels,
                axes = ('b', 0, 1, 'c'))

        self.transformer = Conv2D(filters = self.W_h0,
            batch_size = self.mlp.batch_size,
            input_space = self.input_space,
            output_axes = self.detector_space.axes,
            subsample = self.kernel_stride, 
            border_mode = self.border_mode,
            filters_shape = self.W_h0.get_value(borrow=True).shape)

        W, = self.transformer.get_params()
        W.name = 'W'

        print 'Input shape: ', self.input_space.shape
        print 'Detector space: ', self.detector_space.shape
        
        output_r = (self.detector_space.shape[0] - 1) * \
                self.kernel_stride[0] + self.expand_shape[0]
        output_c = (self.detector_space.shape[1] - 1) * \
                self.kernel_stride[1] + self.expand_shape[1]
        
        self.output_space = Conv2DSpace(shape=[output_r, output_c],
                                    num_channels = 1, 
                                    axes = ('b', 0, 1, 'c'))
        print 'Output space: ', self.output_space.shape

    def get_params(self):
        return []
            
    def fprop(self, state_below):
        # mlp 1st layer, standard conv
        self.input_space.validate(state_below)

        #b = self.b_h0.dimshuffle(0, 'x', 'x', 'x') # ('c',) -> ('c', 0, 1, 'b')

        z = self.transformer.lmul(state_below) + self.b_h0
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        d = z * (z > 0.)
        self.z_h0 = z
        self.d_h0 = d

        self.detector_space.validate(d)
        #assert self.detector_space.num_channels % 16 == 0
        
        # ('c', 0, 1, 'b') -> ('b', 0, 1, 'c')
        #d = d.dimshuffle(3, 1, 2, 0)
        
        # mlp 2nd layer, 1x1 conv        
        #Channel vector at each point is c. 
        #Compute new channel vector c' = max(Wc + b, 0)
#        z = d.dimshuffle(0, 1, 2, 3, 'x') * self.W_h1
#        z = z.sum(axis=3) + self.b_h1
#        d = z * (z > 0.)
#        self.z_h1 = z
#        self.d_h1 = d
        
        # mlp output layer, 1x1 conv
        z = d.dimshuffle(0, 1, 2, 3, 'x') * self.W_y
        z = z.sum(axis=3) + self.b_y
        d = z * (z > 0.)
        self.z_y = z
        self.d_y = d
        
        # expand channel vector as spatial square predictions
        wide_b01c = expand_2d(d, self.expand_shape, self.kernel_stride, 
                  self.detector_space.shape)
        self.output_space.validate(wide_b01c)
        return wide_b01c
    
class PretrainedMLPConvC01B(Layer):
    def __init__(self,
                 kernel_shape,
                 kernel_stride,
                 detector_channels,
                 expand_shape,
                 micro_mlp,
                 layer_name,
                 border_mode = 'valid',
                 pad = 0,
                 partial_sum = 1):

        self.__dict__.update(locals())
        del self.self
        
    def set_input_space(self, space):
        self.input_space = space
        rng = self.mlp.rng
        
        # Make sure number of channels is supported by cuda-convnet
        # (multiple of 4 or <= 3)
        # If not supported, pad the input with dummy channels
        ch = self.input_space.num_channels
        assert ch <= 3 or ch % 4 == 0
        
        params = self.micro_mlp.get_params()   
        self.b_h0 = params[1]
        self.W_h1 = params[2]
        self.b_h1 = params[3]
        self.W_y = params[4]
        self.b_y = params[5]
        W_h0 = params[0].get_value().reshape((self.input_space.num_channels,
                                       self.kernel_shape[0],
                                       self.kernel_shape[1],
                                       self.detector_channels))
        self.W_h0 = sharedX(W_h0, params[0].name)
        
        if self.border_mode == 'valid':
            output_shape = [(self.input_space.shape[0] - self.kernel_shape[0]) / self.kernel_stride[0] + 1,
                (self.input_space.shape[1] - self.kernel_shape[1]) / self.kernel_stride[1] + 1]
        elif self.border_mode == 'full':
            output_shape = [(self.input_space.shape[0] +  self.kernel_shape[0]) / self.kernel_stride[0] - 1,
                    (self.input_space.shape[1] + self.kernel_shape[1]) / self.kernel_stride_stride[1] - 1]

        self.detector_space = Conv2DSpace(shape=output_shape,
                num_channels = self.detector_channels,
                axes = ('c', 0, 1, 'b'))

        print 'type(self.W_h0) =', type(self.W_h0)
        self.transformer = Conv2D(filters = self.W_h0,
            input_axes = self.input_space.axes,
            output_axes = self.detector_space.axes,
            kernel_stride = self.kernel_stride, pad=self.pad,
            message = "", partial_sum=self.partial_sum)

        W, = self.transformer.get_params()
        W.name = 'W'

        print 'Input shape: ', self.input_space.shape
        print 'Detector space: ', self.detector_space.shape
        
        output_r = (self.detector_space.shape[0] - 1) * \
                self.kernel_stride[0] + self.expand_shape[0]
        output_c = (self.detector_space.shape[1] - 1) * \
                self.kernel_stride[1] + self.expand_shape[1]
        
        self.output_space = Conv2DSpace(shape=[output_r, output_c],
                                    num_channels = 1, 
                                    axes = ('b', 0, 1, 'c'))
        print 'Output space: ', self.output_space.shape

    def get_params(self):
        return []
            
    def fprop(self, state_below):
        # mlp 1st layer, standard conv
        self.input_space.validate(state_below)

        b = self.b_h0.dimshuffle(0, 'x', 'x', 'x') # ('c',) -> ('c', 1, 2, 'b')

        z = self.transformer.lmul(state_below) + b
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        d = z * (z > 0.)
        self.z_h0 = z
        self.d_h0 = d

        self.detector_space.validate(d)
        assert self.detector_space.num_channels % 16 == 0
        
        # ('c', 0, 1, 'b') -> ('b', 0, 1, 'c')
        d = d.dimshuffle(3, 1, 2, 0)
        
        # mlp 2nd layer, 1x1 conv        
        #Channel vector at each point is c. 
        #Compute new channel vector c' = max(Wc + b, 0)
        z = d.dimshuffle(0, 1, 2, 3, 'x') * self.W_h1
        z = z.sum(axis=3) + self.b_h1
        d = z * (z > 0.)
        self.z_h1 = z
        self.d_h1 = d
        # mlp output layer, 1x1 conv
        z = d.dimshuffle(0, 1, 2, 3, 'x') * self.W_y
        z = z.sum(axis=3) + self.b_y
        d = z * (z > 0.)
        self.z_y = z
        self.d_y = d
        
        # expand channel vector as spatial square predictions
        wide_b01c = expand_2d(d, self.expand_shape, self.kernel_stride, 
                  self.detector_space.shape)
        self.output_space.validate(wide_b01c)
        return wide_b01c

def expand_2d(b01c, expand_shape, expand_stride, image_shape):
    for b01cv in get_debug_values(b01c):
        assert not np.any(np.isinf(b01cv))
        assert b01cv.shape[1] == image_shape[0]
        assert b01cv.shape[2] == image_shape[1]
        assert b01cv.shape[3] == np.prod(expand_shape)
        
    for i in range(len(expand_shape)):
        assert expand_shape[i] % expand_stride[i] ==0
        
    b0101 = b01c.reshape((b01c.shape[0], image_shape[0], image_shape[1],
                          expand_shape[0], expand_shape[1]))
         
    required_r = (image_shape[0] - 1) * expand_stride[0] + expand_shape[0]
    required_c = (image_shape[1] - 1) * expand_stride[1] + expand_shape[1]
    wide_b01 = T.alloc(0., b01c.shape[0], required_r, required_c)
    
    for row_within_expand in xrange(expand_shape[0]):
        row_stop = (image_shape[0] - 1) * expand_stride[0] + \
                    row_within_expand + 1
        for col_within_expand in xrange(expand_shape[1]):
            col_stop = (image_shape[1] - 1) * expand_stride[1] + \
                        col_within_expand + 1
            wide_b01 = T.inc_subtensor(wide_b01[:,
                row_within_expand:row_stop:expand_stride[0], 
                col_within_expand:col_stop:expand_stride[1]],
            b0101[:,:,:,row_within_expand, col_within_expand])
            
    wide_b01 = wide_b01 / (expand_shape[0] / expand_stride[0]) ** 2
    wide_b01c = wide_b01.reshape((b01c.shape[0], required_r, required_c, 1))
    return wide_b01c

def lwta_2d_b012c(b012c, pool_shape, pool_stride, video_shape):
    """
    Modified from pylearn2.models.mlp.max_pool_c01b.
    """
    mx = None
    t, r, c = video_shape
    pt, pr, pc = pool_shape
    ts, rs, cs = pool_stride
    assert pt == 1
    assert ts == 1
    #assert pt > 0
    assert pr > 0
    assert pc > 0
    #assert pt <= t
    assert pr <= r
    assert pc <= c

    # Compute index in pooled space of last needed pool
    # (needed = each input pixel must appear in at least one pool)
    def last_pool(im_shp, p_shp, p_strd):
        rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
        assert p_strd * rval + p_shp >= im_shp
        assert p_strd * (rval - 1) + p_shp < im_shp
        return rval
    # Compute starting row of the last pool
    #last_pool_t = last_pool(video_shape[0] ,pool_shape[0], pool_stride[0]) * pool_stride[0]
    # Compute number of rows needed in image for all indexes to work out
    required_t = t

    last_pool_r = last_pool(video_shape[1] ,pool_shape[1], pool_stride[1]) * pool_stride[1]
    required_r = last_pool_r + pc
    
    last_pool_c = last_pool(video_shape[2] ,pool_shape[2], pool_stride[2]) * pool_stride[2]
    required_c = last_pool_c + pc

    for b012cv in get_debug_values(b012c):
        assert not np.any(np.isinf(b012cv))
        assert b012cv.shape[1] == t
        assert b012cv.shape[2] == r
        assert b012cv.shape[3] == c

    wide_infinity = T.alloc(-np.inf, 
                            b012c.shape[0],
                            required_t, 
                            required_r, 
                            required_c, 
                            b012c.shape[4])

    name = b012c.name
    if name is None:
        name = 'anon_b012c'
    b012c = T.set_subtensor(wide_infinity[:, 0:t, 0:r, 0:c, :], b012c)
    b012c.name = 'infinite_padded_' + name

    for row_within_pool in xrange(pool_shape[1]):
        row_stop = last_pool_r + row_within_pool + 1
        for col_within_pool in xrange(pool_shape[2]):
            col_stop = last_pool_c + col_within_pool + 1
            cur = b012c[:,
                        :,
                        row_within_pool:row_stop:rs, 
                        col_within_pool:col_stop:cs, 
                        :]
            if mx is None:
                mx = cur
            else:
                mx = T.maximum(mx, cur)
    
    for row_within_pool in xrange(pool_shape[1]):
        row_stop = last_pool_r + row_within_pool + 1
        for col_within_pool in xrange(pool_shape[2]):
            col_stop = last_pool_c + col_within_pool + 1
            cur = b012c[:,
                        :,
                        row_within_pool:row_stop:rs, 
                        col_within_pool:col_stop:cs, 
                        :]
            b012c[:,
                :,
                row_within_pool:row_stop:rs, 
                col_within_pool:col_stop:cs, 
                :] = cur * (cur >= mx)
                        
    for b012cv in get_debug_values(b012c):
        assert not np.any(np.isnan(b012cv))
        assert not np.any(np.isinf(b012cv))

    return b012c

def lwta_3d_b012c(b012c, pool_shape, pool_stride, video_shape):
    """
    Modified from pylearn2.models.mlp.max_pool_c01b.
    """
    mx = None
    t, r, c = video_shape
    pt, pr, pc = pool_shape
    ts, rs, cs = pool_stride
    assert pt > 0
    assert pr > 0
    assert pc > 0
    assert pt <= t
    assert pr <= r
    assert pc <= c

    # Compute index in pooled space of last needed pool
    # (needed = each input pixel must appear in at least one pool)
    def last_pool(im_shp, p_shp, p_strd):
        rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
        assert p_strd * rval + p_shp >= im_shp
        assert p_strd * (rval - 1) + p_shp < im_shp
        return rval
    # Compute starting row of the last pool
    last_pool_t = last_pool(video_shape[0] ,pool_shape[0], pool_stride[0]) * pool_stride[0]
    # Compute number of rows needed in image for all indexes to work out
    required_t = last_pool_t + pr

    last_pool_r = last_pool(video_shape[1] ,pool_shape[1], pool_stride[1]) * pool_stride[1]
    required_r = last_pool_r + pc
    
    last_pool_c = last_pool(video_shape[2] ,pool_shape[2], pool_stride[2]) * pool_stride[2]
    required_c = last_pool_c + pc

    for b012cv in get_debug_values(b012c):
        assert not np.any(np.isinf(b012cv))
        assert b012cv.shape[1] == t
        assert b012cv.shape[2] == r
        assert b012cv.shape[3] == c

    wide_infinity = T.alloc(-np.inf, 
                            b012c.shape[0],
                            required_t, 
                            required_r, 
                            required_c, 
                            b012c.shape[4])

    name = b012c.name
    if name is None:
        name = 'anon_b012c'
    b012c = T.set_subtensor(wide_infinity[:, 0:t, 0:r, 0:c, :], b012c)
    b012c.name = 'infinite_padded_' + name

    for time_within_pool in xrange(pool_shape[0]):
        time_stop = last_pool_t + time_within_pool + 1    
        for row_within_pool in xrange(pool_shape[1]):
            row_stop = last_pool_r + row_within_pool + 1
            for col_within_pool in xrange(pool_shape[2]):
                col_stop = last_pool_c + col_within_pool + 1
                cur = b012c[:,
                            time_within_pool:time_stop:ts,
                            row_within_pool:row_stop:rs, 
                            col_within_pool:col_stop:cs, 
                            :]
                if mx is None:
                    mx = cur
                else:
                    mx = T.maximum(mx, cur)
    
    for time_within_pool in xrange(pool_shape[0]):
        time_stop = last_pool_t + time_within_pool + 1    
        for row_within_pool in xrange(pool_shape[1]):
            row_stop = last_pool_r + row_within_pool + 1
            for col_within_pool in xrange(pool_shape[2]):
                col_stop = last_pool_c + col_within_pool + 1
                cur = b012c[:,
                            time_within_pool:time_stop:ts,
                            row_within_pool:row_stop:rs, 
                            col_within_pool:col_stop:cs, 
                            :]
                b012c = T.set_subtensor(b012c[:,
                    time_within_pool:time_stop:ts,
                    row_within_pool:row_stop:rs, 
                    col_within_pool:col_stop:cs, 
                    :], cur * (cur >= mx))
       
    b012c = b012c[:, 0:t, 0:r, 0:c, :] # remove infinity padding                 
    for b012cv in get_debug_values(b012c):
        assert not np.any(np.isnan(b012cv))
        assert not np.any(np.isinf(b012cv))

    return b012c

"""
class Conv3DLWTA(Conv3DLinear):
    def __init__(self,
                 output_channels,
                 kernel_shape,
                 layer_name,
                 irange,
                 lwta_shape,
                 weights = None,
                 bias = None,
                 border_mode = 'valid',
                 #sparse_init = None,
                 include_prob = 1.0,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 left_slope = 0.0,
                 tied_b = True,
                 max_kernel_norm = None,
                 pool_type = 'max',
                 detector_normalization = None,
                 output_normalization = None,
                 kernel_stride=(1, 1, 1)):

        self.__dict__.update(locals()) 
        del self.self

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if not hasattr(self, 'tied_b'):
            self.tied_b = False
        if self.tied_b:
            b = self.b.dimshuffle('x', 'x', 0, 'x', 'x') # ('c',) -> ('b', 0, 'c', 1, 2)
        else:
            b = self.b.dimshuffle('x', 0, 1, 2, 3)    # (0, 'c', 1, 2) -> ('b', 0, 'c', 1, 2)

        z = self.transformer.lmul(state_below) + b
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        d = z * (z > 0.) + self.left_slope * z * (z < 0.)
        self.detector_space.validate(d)
        
        if np.prod(self.lwta_shape) == 1:
            return d
        
        # ('b', 0, 'c', 1, 2) -> ('b', 0, 1, 2, 'c')
        d = d.dimshuffle(0, 1, 3, 4, 2)
        w = lwta_3d_b012c(d, 
                   self.lwta_shape, 
                   self.lwta_shape, 
                   self.output_space.shape)
        
        # ('b', 0, 1, 2, 'c') -> ('b', 0, 'c', 1, 2)
        w = w.dimshuffle(0, 1, 4, 2, 3)
        self.output_space.validate(w)
        return w

class TransformConv3D(Layer):

    def __init__(self, 
                 output_channels,
                 layer_name, 
                 lwta_shape,
                 irange = None,
                 istdev = None,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None):
        
        self.__dict__.update(locals())
        del self.self
        #self._params = []

    def set_input_space(self, space):
        self.input_space = space
        if self.output_channels is None:
            self.output_space = self.input_space
            return 
        
        self.output_space = Conv3DSpace(shape=self.input_space.shape,
            num_channels = self.output_channels,
            axes = ('b', 0, 'c', 1, 2))
    
        rng = self.mlp.rng
        if self.irange is not None:
            assert self.istdev is None
            W = rng.uniform(-self.irange, self.irange,
                        (self.input_space.num_channels, self.output_channels))
        else:
            assert self.istdev is not None
            W = rng.randn(self.input_space.num_channels, self.output_channels) * self.istdev
        W = sharedX(W)
        W.name = self.layer_name + '_W'    
        self.W = W       
        self.b = sharedX(np.zeros((self.output_channels,)) + self.init_bias, 
                         name = self.layer_name + '_b')                     
                                        
    def get_params(self):
        if self.output_channels is None:
            return []
        rval = [self.W, self.b]
        return rval

    def get_monitoring_channels(self):
        #return OrderedDict()
        W = self.W

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))
        #col_norms_stdev = T.sqrt(T.sqr(col_norms - col_norms.mean()).mean())

        return OrderedDict([
                            ('col_norms_min'  , col_norms.min()),
                            ('col_norms_mean' , col_norms.mean()),
                            ('col_norms_max'  , col_norms.max()),
                            #('col_norms_stdev'  , col_norms_stdev),
                            ])

    def get_monitoring_channels_from_state(self, state, target=None):
        return  OrderedDict()
            
    def fprop(self, state_below):
        self.input_space.validate(state_below)
        
        if self.output_channels is None:
            d = state_below.dimshuffle(0, 1, 3, 4, 2)
        else:
            batch_size = state_below.shape[0]
            # ('b', 0, 'c', 1, 2) -> ('b', 0, 1, 2, 'c', 'x')
            p = state_below.dimshuffle(0, 1, 3, 4, 2, 'x')
            #Channel vector at each point is c. 
            #Compute new channel vector c' = max(Wc + b, 0)
            z = p * self.W
            z = z.sum(axis=4) + self.b
            d = z * (z > 0.)
        
        if np.prod(self.lwta_shape) == 1:
            w = d
        else:
            w = lwta_3d_b012c(d, 
                       self.lwta_shape, 
                       self.lwta_shape, 
                       self.output_space.shape)
        
        # ('b', 0, 1, 2, 'c') -> ('b', 0, 'c', 1, 2)
        return w.dimshuffle(0, 1, 4, 2, 3)

"""
class FlattenConv3D(Layer):
    """
    First do convolution with 1x1x1 kernel to change the number of channels, 
    as mlpconv in NIN paper.
    Then apply LWTA within each channel.
    """

    def __init__(self, 
                 num_channel,
                 layer_name, 
                 irange = None,
                 istdev = None,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None):
        
        self.__dict__.update(locals())
        del self.self
        self.b = sharedX(np.zeros((self.num_channel,)) + init_bias, 
                         name = layer_name + '_b')
        #self._params = []

    def set_input_space(self, space):
        self.input_space = space
        self.output_space = VectorSpace(np.prod(self.input_space.shape) *
                                        self.num_channel)
        
        rng = self.mlp.rng
        if self.irange is not None:
            assert self.istdev is None
            W = rng.uniform(-self.irange, self.irange,
                        (self.input_space.num_channels, self.num_channels))
        else:
            assert self.istdev is not None
            W = rng.randn(self.input_space.num_channels, self.num_channels) * self.istdev
        W = sharedX(W)
        W.name = self.layer_name + '_W'                                
                                        
    def get_params(self):
        rval = [self.W, self.b]
        return rval

    def get_monitoring_channels(self):
        #return OrderedDict()
        W = self.W

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))
        #col_norms_stdev = T.sqrt(T.sqr(col_norms - col_norms.mean()).mean())

        return OrderedDict([
                            ('col_norms_min'  , col_norms.min()),
                            ('col_norms_mean' , col_norms.mean()),
                            ('col_norms_max'  , col_norms.max()),
                            #('col_norms_stdev'  , col_norms_stdev),
                            ])

    def get_monitoring_channels_from_state(self, state, target=None):
        return  OrderedDict()
            
    def fprop(self, state_below):
        self.input_space.validate(state_below)
        batch_size = state_below.shape[0]
        
        # ('b', 0, 'c', 1, 2) -> ('b', 'c', 0, 1, 2) -> ('b', 'c', v)
        p = state_below.dimshuffle(0, 2, 1, 3, 4).reshape((
                                            batch_size, 
                                            self.input_space.get_total_dimension()))
        #Channel vector at each point is c. 
        #Compute new channel vector c' = max(Wc + b, 0)
        z = p.dimshuffle(0, 1, 2, 'x') * self.W
        z = z.sum(axis=2) + self.b
        d = z * (z > 0.)
        
        # d has shape (b, v, c'), now do LWTA
        channel_max = d.max(axis=1).dimshuffle(0, 'x', 1) * T.ones_like(d)
        w = d * (d >= channel_max)
        return w.reshape((batch_size, self.num_channels *
                                    np.prod(self.input_space.shape)))

from pylearn2.costs.cost import Cost
from pylearn2.space import CompositeSpace
class Baseline(Cost):

    supervised = True

    def __init__(self):
        #self.__dict__.update(locals())
        #del self.self
        pass

    def expr(self, model, data, ** kwargs):
        space, sources = self.get_data_specs(model)
        space.validate(data)
        (X, Y) = data
        input_space = model.get_input_space()
        output_space = model.get_output_space()
        X = Conv2DSpace.convert(X, input_space.axes, output_space.axes)
        Y = Y.reshape((Y.shape[0], 
                       output_space.num_channels, 
                       output_space.shape[0], 
                       output_space.shape[1]))
        Y_hat = X[:, -1, 
            (input_space.shape[0]-output_space.shape[0])/2:
            (input_space.shape[0]-output_space.shape[0])/2+output_space.shape[0], 
            (input_space.shape[1]-output_space.shape[1])/2:
            (input_space.shape[1]-output_space.shape[1])/2+output_space.shape[1],
            ]
        Y_hat = Y_hat.dimshuffle(0, 'x', 1, 2)
        return T.sqr(Y - Y_hat).sum(axis=(1,2,3)).mean()

    def get_data_specs(self, model):
        space = CompositeSpace([model.get_input_space(), model.get_output_space()])
        sources = (model.get_input_source(), model.get_target_source())
        return (space, sources)