# -*- coding: utf-8 -*-
"""
@author: Louis Lettry (lettryl@vision.ee.ethz.ch)
"""


import tensorflow as tf
tf.reset_default_graph()    #   Clear tensorflow <-- specially useful when running in permanent environment

import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt  

from net_model import make_net


#   Util to convert image to the CNN format
def CNNify(img):
    img = img.astype(dtype=np.float) / 255.
    return img[None,:,:,:]
    
#   Util to revert back from the CNN format to image
def unCNNify(img):
    if len(img.shape) == 4:
        img = img[0,:]
        
    return np.clip(img*255., 0, 255).astype(dtype=np.uint8)

def show(name, img):
    fig = plt.figure(name)
    ax = fig.add_subplot(111)
    ax.imshow(img)

if __name__ == "__main__":
    plt.close("all")

    #   Can be set to (320, 240) to use training dimensions
    input_size = (None, None)
    input_img = tf.placeholder(tf.float32,shape=(1, input_size[0], input_size[1], 3), name = "I")
    is_training = tf.placeholder(tf.bool, name="is_training")    
    
    #   Create network
    net = make_net(input_img, is_training)
    fetches = [net["A"], net["S"]]

    # setup session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess=tf.InteractiveSession(config=config)
    
    #   Utils to restore weights
    def restore_map_vars(model_checkpoint_path):
        reader = tf.train.NewCheckpointReader(model_checkpoint_path)
        saved_shapes = reader.get_variable_to_shape_map()    
        var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables() if var.name.split(':')[0] in saved_shapes])
        restore_vars = {}
        name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
        
        #   Map name to var
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                curr_var = name2var[saved_var_name]
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars[saved_var_name] = curr_var
                else:
                    print "Not found =", saved_var_name
                    return None
        
        #   Reminiscences from author's aging framework
        restore_vars["S/w"] = name2var["S/kernel"]
        restore_vars["S/b"] = name2var["S/bias"]
                    
        return restore_vars    
    
    #   Load trained weights
    pretrained_path = "model/model_params.ckpt"  
    common_vars = restore_map_vars(pretrained_path)
    saver = tf.train.Saver(var_list = common_vars)
    saver.restore(sess, pretrained_path)
    
    #   Load image
    img_path = "light01.png"
    img = imread(img_path)
    show("Input Image", img)
    img = CNNify(img)
    
    #   Process image
    results = sess.run([net["A"], net["S"]], {input_img:img, is_training:False})
    
    A = unCNNify(results[0])
    S = unCNNify(results[1])
    
    #   Present A
    show("Albedo", A)
    
    #   S € [0, 256] --> rescale to [0,1]
    nS = S/np.max(S)
    show("Shading", S)