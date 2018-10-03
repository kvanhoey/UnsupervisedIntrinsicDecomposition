# -*- coding: utf-8 -*-
"""
@author: Louis Lettry (lettryl@vision.ee.ethz.ch)
"""

import tensorflow as tf

from common.tensorflow.utils import *

def make_net(input_img, is_training):
    
    inception_branches = [(32, 5, 64)]
    nbr_levels = 5
    
    #   make a simple 1x1 projection followed by a standard convolution
    def inception_branch(bid, input_branch, branch):
        name = str(branch[1]) +"x"+str(branch[1])
        with tf.name_scope("branch"+name) as scope:
            #   First 1x1 projection
            if branch[0] > 0:
                conv1x1_pre = tf.layers.conv2d(input_branch, branch[0], (1, 1), name = "conv1x1_pre" + name + "_" + str(bid), padding = "same")
                conv1x1_pre = tf.layers.batch_normalization(conv1x1_pre, training=is_training, name = "conv1x1_pre" + name + "_bn"+str(bid))
                conv1x1_pre = tf.nn.elu(conv1x1_pre, name = "conv1x1_pre" + name + "_elu_"+str(bid))
                input_branch = conv1x1_pre
            
            #   Then std convolution
            conv = tf.layers.conv2d(input_branch, branch[2], (branch[1], branch[1]), name = "conv" + name + "_" + str(bid), padding = "same")
            conv = tf.layers.batch_normalization(conv, training=is_training, name = "conv" + name + "_bn"+str(bid))
            conv = tf.nn.elu(conv, name = "conv" + name + "_elu_"+str(bid))
            
            return conv
            
    #   makes block of inception branches
    def inception_block(lid, input_layer):
        with tf.name_scope("inception_"+str(lid)) as scope:
            branches = []
            for branch in inception_branches:
                branches.append(inception_branch(lid+"branch", input_layer, branch))
    
            return tf.concat(branches, axis = 3)
            
    #   recursive skip encoder level
    def skip_encoder_level(lid, input_layer):    
        #   Downward side of the pyramid
        layer = inception_block("down_"+str(lid), input_layer)        

        if lid-1 > 0: 
            level_input = layer
            shape = tf.shape(level_input)[1:3]
            
            #   Downscale
            layer = tf.layers.max_pooling2d(level_input, (3,3), (2,2), name = "down_maxpool_"+str(lid))
            
            #   Next level
            layer = skip_encoder_level(lid-1, layer)
            
            #   Upscale
            layer = tf.image.resize_images(layer, shape)
            
            #   Skip connection
            layer = tf.concat((layer, level_input), axis = 3)
        
        #   Upside of the pyramid
        layer = inception_block("up_"+str(lid), layer)  
        
        return layer
        
    #   make the inception U-Net
    layer = skip_encoder_level(nbr_levels, input_img)

    #   Use division trick
    with tf.name_scope("AS_divtrick") as scope:
        #   Final convolution for shading
        l = tf.layers.conv2d(layer, 3, (1, 1), name = "S", padding = "same")
            
        #   Due to 8bits quantization no need to clip for infinity as inverse will never be smaller than 1/256 (0 excepted)
        S = tf.clip_by_value(l, 1e-5, 256)
        
        l = input_img[:,:,:,0:3] / (S)
        A = tf.clip_by_value(l, 0, 1)
    
    return {"A":A, "S":S}


