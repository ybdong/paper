####待修改

caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)

import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
from pylab import *
import os

full_dataset = False
if full_dataset:
    NUM_STYLE_IMAGES = NUM_STYLE_LABELS = -1
else:
    NUM_STYLE_IMAGES = 200
    NUM_STYLE_LABELS = 5
style_label_file = caffe_root + 'examples/finetune_flickr_style/style_names.txt'
style_labels = list(np.loadtxt(style_label_file, str, delimiter='\n'))
if NUM_STYLE_LABELS > 0:
    style_labels = style_labels[:NUM_STYLE_LABELS]
weights = os.path.join(caffe_root, 'models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel')
assert os.path.exists(weights)
from caffe import layers as L
from caffe import params as P

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param   = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def caffenet(data, label=None, train=True, num_classes=1000,
             classifier_name='fc8', learn_all=False):
    """Returns a NetSpec specifying VGG16Net, following the original proto text
       specification (./models/VGG_ILSVRC_16_layers/train_val.prototxt)."""
    n = caffe.NetSpec()
    n.data = data
    param = learned_param if learn_all else frozen_param
    n.conv1_1, n.relu1_1 = conv_relu(n.data,ks=3, nout=64, pad=1,param=param)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, ks=3 ,nout=64, pad=1,param=param)
    n.pool1 = max_pool(n.relu1_2, ks=2, stride=2)
    n.conv2_1, n.relu2_1 = conv_relu(n.pool1,ks=3,  nout=128, pad=1, param=param)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, ks=3, nout=128, pad=1,param=param)
    n.pool2 = max_pool(n.relu2_2, ks=2, stride=2)
    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, ks=3, nout=256, pad=1,param=param)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, ks=3, nout=256, pad=1,param=param)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, ks=3, nout=256, pad=1,param=param)
    n.pool3 = max_pool(n.relu3_3, ks=2, stride=2)
    n.conv4_1, n.relu4_1 = conv_relu(n.pool3,  ks=3, nout=512, pad=1,param=param)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, ks=3, nout=512, pad=1,param=param)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, ks=3, nout=512, pad=1,param=param)
    n.pool4 = max_pool(n.relu4_3, ks=2, stride=2)
    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, ks=3, nout=512, pad=1,param=param)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, ks=3, nout=512, pad=1,param=param)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, ks=3, nout=512, pad=1,param=param)
    n.pool5 = max_pool(n.relu5_3, ks=2, stride=2)  
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6
    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7
    # always learn fc8 (param=learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)
    if not train:
        n.probs = L.Softmax(fc8)
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc = L.Accuracy(fc8, n.label)
    # write the net to a temporary file and return its filename
    with open('../models/VGG_ILSVRC_16_layers/train_val.prototxt', "w")as f:
        f.write(str(n.to_proto()))
        return f.name
def style_net(train=True, learn_all=False, subset=None):
    if subset is None:
        subset = 'train' if train else 'test'
    source = caffe_root + 'data/bird_classification/%s.txt' % subset
    transform_param = dict(mirror=train, crop_size=224,
        mean_file=caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto')
    style_data, style_label = L.ImageData(
        transform_param=transform_param, source=source,
        batch_size=50, new_height=256, new_width=256, ntop=2)
    return caffenet(data=style_data, label=style_label, train=train,
                    num_classes=NUM_STYLE_LABELS,
                    classifier_name='fc8_birdrecog',
                    learn_all=learn_all)
from caffe.proto import caffe_pb2

def solver(train_net_path, test_net_path=None, base_lr=0.001):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 1000  # Test after every 1000 training iterations.
        s.test_iter.append(100) # Test on 100 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1
    
    s.max_iter = 100000     # # of times to update the net (training iterations)
    
    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 20000

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 10000
    s.snapshot_prefix = caffe_root + 'models/finetune_bird_recognition/bird_recognition'
    
    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    
    # Write the solver to a temporary file and return its filename.
    with open('../models/VGG_ILSVRC_16_layers/solver.prototxt', "w") as f:
        f.write(str(s))
        return f.name