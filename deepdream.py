#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb

# boilerplate code
from __future__ import print_function

import os
import sys
import struct
import cv2
import math
import time
import argparse

import numpy as np
import numpy.random as rnd
from functools import partial

from random import shuffle

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework.dtypes import DType

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='models/tensorflow_inception_graph.pb', help="location of the model to load")
parser.add_argument("--mode", required=True, choices=["graph", "naive", "multiscale", "lapnorm", "dream", "export_tensor_names"])
parser.add_argument("--input_dir", help="path to folder containing images", default="images")
parser.add_argument("--input_image", help="image to dream with", default="000003.png") # can be 'noise' special value
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--output_filetype", default="jpg", choices=["png", "jpg"])
parser.add_argument("--export_tensor_names", type=bool, default=False, help="Export a list of all dreamable tensors as JSON")

# use in dream mode
parser.add_argument("--shuffle", default='no',  choices=["yes", "no"])
parser.add_argument("--iterations", type=int, default=10)
parser.add_argument("--step", type=float, default=1.5)
parser.add_argument("--octaves", type=int, default=4)
parser.add_argument("--octave_scale", type=float, default=1.4)
parser.add_argument("--render_list", default=None, help="Render all listed layers/channels from a list. Expects a text file with each entry on a new line - `tensorname-#channel`")
parser.add_argument("--renders_per_layer", type=int, default=3, help="Number of channels to render from a layer. Set to zero to render all channels")
parser.add_argument("--layer_name", default=None, help="Render channels from a single layer (use --export_tensor_names to get a list of all dreamable tensors)")
parser.add_argument("--layer_names", default=None, help="Render channels from a list of layers. Expects a text file with each entry on a new line (use --export_tensor_names to get a list of all dreamable tensors)")
parser.add_argument("--channel_number", default=-1, type=int, help="Specify a single channel to render. Only applicable when --layer_name is set")
parser.add_argument("--resume_from_layer", default=None, help="Resume a previous deepdream sequence from a given layer (use --export_tensor_names to get a list of all dreamable tensors)")
parser.add_argument("--resume_from_channel", default=-1, type=int, help="Resume a previous deepdream sequence from a given channel of a layer. Only applies when --layer_name or --resume_from_layer are set")

parser.add_argument("--seed", default=123456, type=int)
a = parser.parse_args()

a.seed = a.seed if a.seed>0 else int(time.time())
rnd.seed(a.seed)

# ####################################################
# Loading and displaying the model graph
# ####################################################

#The pretrained network can be downloaded here. Unpack the tensorflow_inception_graph.pb file from the archive and set its path to model_fn variable. Alternatively you can uncomment and run the following cell to download the network:
#!wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip && unzip inception5h.zip


model_fn = a.model #'inception5h\tensorflow_inception_graph.pb'

# creating TensorFlow session and loading the model
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())

inputName = "image"# input tensor names - inception v3:"image",  inception_v4:"InputImage"

# define the input tensor
t_input = tf.placeholder(np.float32, name='input')
imagenet_mean = 117.0 # ? how is this value determined (and what does it do)
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})

# Helper functions for TF Graph visualization

def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = tf.compat.as_bytes("<stripped %d bytes>"%size)
    return strip_def

#
def save_graph(graph_def, max_const_size=32):

    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()

    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    txt_out = repr(str(strip_def))
    txt_out = txt_out.replace("\\n","\n")

    with open("graph.txt", "w") as text_file:
        text_file.write(txt_out)

#
def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add()
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
    return res_def


# ####################################################
# Naive feature visualisation
# ####################################################
current_layer=None
current_channel=0
# start with a gray image with a little noise
img_noise = rnd.uniform(size=(224,224,3)) + 100.0

def show_array(arr):
    img = np.uint8(np.clip(arr, 0, 1)*255)
    cv2.imshow('dst_rt', img)

# float array in, rgb uint8 out
def save_array(arr, name, ext="png"):
    img = np.uint8(np.clip(arr, 0, 1)*255)
    cv2.imwrite("{d}/{n}.{e}".format(d=a.output_dir, n=name, e=ext), img)

def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0"%layer)

def render_naive(t_obj, img0=img_noise, iter_n=20, step=1.0):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!

    img = img0.copy()
    for i in range(iter_n):
        g, score = sess.run([t_grad, t_score], {t_input:img})
        # normalizing the gradient, so the same step size should work
        g /= g.std()+1e-8         # for different layers and networks
        img += g*step
        print(score, end = ' ')

    show_array(visstd(img))


# ####################################################
# Multiscale image generation
# ####################################################

def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)


def calc_grad_tiled(img, t_grad, tile_size=512):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over
    multiple iterations.'''
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = rnd.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

##

def render_multiscale(t_obj, img0=img_noise, iter_n=10, step=1.0, octave_n=3, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!

    img = img0.copy()
    for octave in range(octave_n):
        if octave>0:
            hw = np.float32(img.shape[:2])*octave_scale
            img = resize(img, np.int32(hw))
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            # normalizing the gradient, so the same step size should work
            g /= g.std()+1e-8         # for different layers and networks
            img += g*step
            #print('.', end = ' ')

        #show_array(visstd(img))
        save_array(visstd(img), 'multiscale_ocatave_{:d}'.format(octave), a.output_filetype)


# ####################################################
# Laplacian Pyramid Gradient Normalization
# ####################################################

k = np.float32([1,4,6,4,1])
k = np.outer(k, k)
k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)

def lap_split(img):
    '''Split the image into lo and hi frequency components'''
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1,2,2,1])
        hi = img-lo2
    return lo, hi

def lap_split_n(img, n):
    '''Build Laplacian pyramid with n splits'''
    levels = []
    for i in range(n):
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]

def lap_merge(levels):
    '''Merge Laplacian pyramid'''
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
    return img

def normalize_std(img, eps=1e-10):
    '''Normalize image by making its standard deviation = 1.0'''
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img/tf.maximum(std, eps)

def lap_normalize(img, scale_n=4):
    '''Perform the Laplacian pyramid normalization.'''
    img = tf.expand_dims(img,0)
    tlevels = lap_split_n(img, scale_n)
    tlevels = list(map(normalize_std, tlevels))
    out = lap_merge(tlevels)
    return out[0,:,:,:]

#
#
def render_lapnorm(name, t_obj, img0=img_noise, visfunc=visstd,
                   iter_n=10, step=1.0, octave_n=3, octave_scale=1.4, lap_n=4):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    # build the laplacian normalization graph
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))

    img = img0.copy()
    for octave in range(octave_n):
        if octave>0:
            hw = np.float32(img.shape[:2])*octave_scale
            img = resize(img, np.int32(hw))
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            g = lap_norm_func(g)
            img += g*step
            print('.', end = ' ')

        save_array(visfunc(img), name + '_lapnorm_oct-{:d}'.format(octave), a.output_filetype)


# ####################################################
# DeepDream
# ####################################################
def now(): return int(time.time())

def render_deepdream(output_dir, name, t_obj, img0=img_noise,
                     iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):

    fname = output_dir + "/" + name + "_octave-{:d}".format(octave_n)
    savename = "{d}/{n}.{e}".format(d=a.output_dir, n=fname, e=a.output_filetype)

    if os.path.exists(savename):
        print("skip existing " + fname)
        return

    print("Rendering {a}/{b} ({c:d} octaves)".format(a=output_dir, b=name,c=octave_n))
    
    
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!


    # split the image into a number of octaves
    img = img0
    octaves = []
    for i in range(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        hi = img-resize(lo, hw)
        img = lo
        octaves.append(hi)

    # generate details octave by octave
    for octave in range(octave_n):
        if octave>0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2])+hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g*(step / (np.abs(g).mean()+1e-7))
            #print('.',end = ' ')

    print("Saving:", fname)
    save_array(img/255.0, fname, a.output_filetype)


#
#
def dreamable_shape(name):

    test = None

    # example with channel number - conv2d2_pre_relu-051
    if name.index("-") > -1:
        test = name[:-4]
        #print("layer has channel number " + test)
        #print(name[-3:]) # channel num - always 3 digits
    else:
        test = name

    # to find layers to deepdream with we want to match ones with the same shapes/dims as the examples (?,?,?,int)
    target_shape = tf.TensorShape([None, None, None, 512])# T("mixed4d_3x3_bottleneck_pre_relu").shape
    target_dims = len(target_shape)

    # add layer names here to ignore them while dreaming
    ignore_layers = ["avgpool0"] # processing avgpool0 generates an error ("computed output size would be negative")

    if test in ignore_layers: return False

    layer = T(test)

    if layer.shape.dims is None: return False
    if not len(layer.shape) is target_dims: return False

    for x in range(0, target_dims):
        if not type(layer.shape[x].value) is type(target_shape[x].value):
            return False

    return True


# ####################################################
# Main
# ####################################################

# collect and filter tensor names so we only attempt to dream with ones that have the required shape/dims
def setup_layers():
    names = None
    
    if a.render_list is not None:
        print("loading render_list from file:", a.render_list)
        with open(a.render_list) as f:
            lines = f.read().splitlines()
            names = [n for n in lines if dreamable_shape(n)]
            if a.shuffle == "yes": shuffle(names)


    elif a.layer_names is not None:
        # use a custom list of tensor names
        print("loading layer names from file:", a.layer_names)
        with open(a.layer_names) as f:
            lines = f.read().splitlines()
            names = [n for n in lines if dreamable_shape(n)]
            if a.shuffle == "yes": shuffle(names)
    else:
        # use all of the dreamable tensors in the graph
        names = [n.name for n in graph_def.node if dreamable_shape(n.name)]


    print('Loaded model:', model_fn)

    if a.render_list is None:
        feature_counts = [int(T(n).get_shape()[-1]) for n in names]
        feature_count = sum(feature_counts)

        print('layers:', len(names))
        print('feature channels:', feature_count)
    else:
        print('features to render:', len(names))

    return names


def main():

    # names of all the layers we're going to dream with
    dreamy_tensors = setup_layers()


    if(a.mode == "graph"):
        print("Dumping model graph...")
        # A dump of the graph structure. We are going to visualize "Conv2D" nodes.
        tmp_def = rename_nodes(graph_def, lambda s:"/".join(s.split('_',1)))
        save_graph(tmp_def)

    elif a.mode == "export_tensor_names":
        # slightly different to graph mode - just exports the list of all dreamable tensor names
        model_name = model_fn.split("\\")[-1]
        with gfile.FastGFile("{dir}/{model}-tensors.json".format(dir=a.output_dir, model=model_name), 'wb') as f:
            f.write(str(dreamy_tensors).replace("'","\""))

    else:

        if(a.mode == "naive"):
            print('naive')
            #cv2.waitKey(0)
            render_naive(T('mixed4d_3x3_bottleneck_pre_relu')[:,:,:,139])
            cv2.waitKey(0)

        elif(a.mode == "multiscale"):
            print('multiscale')
            render_multiscale(T('mixed4d_3x3_bottleneck_pre_relu')[:,:,:,480])
            cv2.waitKey(0)

        elif(a.mode == "lapnorm"):
            layer_name = "mixed4d_3x3_bottleneck_pre_relu"
            render_lapnorm("c"+str(channel), T(layer_name)[:,:,:,channel])

            render_lapnorm("e7_c16", T(layer_name)[:,:,:,16])
            render_lapnorm("e7_c139", T(layer_name)[:,:,:,139])
            render_lapnorm("e7_c255", T(layer_name)[:,:,:,255])
            render_lapnorm("e7_c480", T(layer_name)[:,:,:,480])
            render_lapnorm("e7_c511", T(layer_name)[:,:,:,511])

            # Lower layers produce features of lower complexity.
            render_lapnorm("e3_c101", T(layer_name)[:,:,:,101])

            # There are many interesting things one may try. For example, optimizing a linear combination of features often gives a "mixture" pattern.
            render_lapnorm("d", T(layer_name)[:,:,:,65]+T(layer_name)[:,:,:,139], octave_n=4)

            cv2.waitKey(0)

        elif(a.mode == "dream"):

            source_image = a.input_image

            # dream settings
            dd_iterations = a.iterations #10
            dd_step = a.step #1.5
            dd_octaves = a.octaves#3
            dd_octave_scale = a.octave_scale#1.4
            renders_per_layer = a.renders_per_layer
            if renders_per_layer<0: renders_per_layer=0
            # manually set this to skip into the process and resume rendering dreams where you left off
            start_tensor = a.resume_from_layer #"mixed3a_3x3_bottleneck_pre_relu"
            start_channel = a.resume_from_channel

            # setup output directories
            image_dir = source_image.split(".")[0]
            dest_dir = a.output_dir + "/" + image_dir
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            # prepare the input image
            if(source_image == "noise"):
                # img0 = rnd.uniform(size=(1080,1920,3))
                img0 = rnd.uniform(size=(512,512,3))
            else:
                img0 = cv2.imread(a.input_dir + "/" + a.input_image)

            img0 = np.float32(img0)


            if a.render_list is not None:
                # just render all listed items - names should include channel number at the end `{tensorname}-###`
                n = len(dreamy_tensors)
                print("rendinging {n:d} features)".format(n=n))

                for i in range(n):
                    tn = dreamy_tensors[i][:-4] # trim channel num from end of tensor name
                    channel = int(dreamy_tensors[i][-3:]) # channel num - always the last 3 digits
                    
                    layer = T(tn) 
                    tnsr = tn.replace("/","__")

                    print("Processing {i:d}/{n:d} {tn})".format(i=i,n=n,tn=tn))

                    dd_name = '{tname}-{channel:03d}'.format(tname=tnsr, channel=channel)
                    render_deepdream(image_dir, dd_name, tf.square(layer[:,:,:,channel]), img0, dd_iterations, dd_step, dd_octaves, dd_octave_scale)

                print("done")
                return
                


            render_count=0
            channel_skip=0
            skip_count=0

            layer_count = len(dreamy_tensors)
            layer_start = 0

            if start_tensor is not None:
                if start_tensor in dreamy_tensors:
                    layer_start = dreamy_tensors.index(start_tensor)
                else:
                    print("Layer {a} does not exist".format(a=layer_start))
                    exit()

            # rendering a single, specific layer?
            if a.layer_name is not None:
                layer_start = dreamy_tensors.index(a.layer_name)
                if layer_start < 0:
                    print("Layer {a} does not exist".format(a=a.layer_name))
                    exit()
                else:
                    print("Rendering from layer {a}".format(a=a.layer_name))
                    layer_count = layer_start + 1
            elif start_tensor is not None:
                print("Resuming render from {t} (layer {n}/{k})".format(t=start_tensor, n=layer_start, k=layer_count))


            channel_count = int(T(dreamy_tensors[layer_start]).shape[3])
            if start_channel > -1 and start_channel < channel_count:
                print("Resuming at channel ", start_channel)
            else:
                start_channel = -1

            for i in range(layer_start, layer_count):
                
                tn = dreamy_tensors[i]

                layer = T(tn)
                tnsr = tn.replace("/","__")

                n=0
                channel_count = int(layer.shape[3])
                renders_per_layer = a.renders_per_layer
                if renders_per_layer > channel_count or renders_per_layer <= 0: renders_per_layer = channel_count

                if a.channel_number >= 0:
                    # render from a single channel
                    if a.channel_number > channel_count:
                        print("channel_number out of range. Layer {l} has {n} channels".format(l=tn, n=channel_count))
                        exit()

                    n = a.channel_number
                    channel_skip = channel_count
                    channel_count = 1

                else:
                    # render multiple channels from the layer. uses renders_per_layer
                    channel_skip = channel_count / renders_per_layer
                    if(channel_skip <= 0): channel_skip = 1

                    n = int(channel_count - (channel_skip/2))
                    if n == channel_count: n = n-1


                print("Processing {i:d}/{n:d} {tn} ({cc:d} channels, renders_per_layer:{rpl:d} channel_skip:{cskp})"
                .format(i=i,n=layer_count,tn=tn,cc=channel_count,rpl=renders_per_layer,cskp=channel_skip))

                if start_channel > -1 : n = start_channel

                while (n >= 0):
                    dd_name = '{tname}-{channel:03d}'.format(tname=tnsr, channel=n)
                    render_deepdream(image_dir, dd_name, tf.square(layer[:,:,:,n]), img0, dd_iterations, dd_step, dd_octaves, dd_octave_scale)
                    n = round(n - channel_skip)

                    # todo - try other operations on the tensor(s) and combine more than one - https://www.tensorflow.org/api_docs/python/tf/square
                    # There are many interesting things one may try. For example, optimizing a linear combination of features often gives a "mixture" pattern.
                    # dream_tensor = T(layer_name)[:,:,:,65]+T(layer_name)[:,:,:,139]

                start_channel=-1

            print("Dreaming complete. Wake up.")

main()
