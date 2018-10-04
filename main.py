# -*- coding: utf-8 -*-
"""
Code for executing the Single-Image Intrinsic Decomposition algorithm presented in the paper:
'Unsupervised Deep Single-Image Intrinsic Decomposition using Illumination-Varying Image Sequences'
by Louis Lettry (lettryl@vision.ee.ethz.ch), Kenneth Vanhoey (kenneth@research.kvanhoey.eu) and Luc Van Gool
in Computer Graphics Forum, vol. 37, no. 10 (Proceedings of Pacific Graphics 2018)

Code author: Louis Lettry

Usage: This file decomposes all images in the folder dir_in (see parameters just below) into Albedo and Shading using our pre-trained CNN.
"""

### Global parameters ###
dir_in = "input/" # Folder containing all images
valid_images = [".jpg",".gif",".png",".tga",".tif"] # Valid extensions, feel free to extend
dir_out = "output/" # Folder to save decompositions in

### Imports and prep ###
import os
if not os.path.exists(dir_out): # Create output dir
    os.makedirs(dir_out)

import numpy as np

from PIL import Image

import tensorflow as tf
tf.reset_default_graph()    #   Clear tensorflow <-- specially useful when running in permanent environment

from net_model import make_net

### Utility functions ###
#   image reading function
def imread(path):
    img = Image.open(path)
    return np.array(img)

#   image writing function
def imwrite(path, img):
    img = Image.fromarray(img)
    img.save(path)

#   Util to convert image to the CNN format
def CNNify(img):
    img = img.astype(dtype=np.float) / 255.
    return img[None,:,:,:]

#   Util to revert back from the CNN format to image
def unCNNify(img):
    if len(img.shape) == 4:
        img = img[0,:]

    return np.clip(img*255., 0, 255).astype(dtype=np.uint8)

### Main ###
if __name__ == "__main__":
    input_size = (None, None) #   Can be set to (320, 240) to use training dimensions
    input_img = tf.placeholder(tf.float32,shape=(1, input_size[0], input_size[1], 3), name = "I")
    is_training = tf.placeholder(tf.bool, name="is_training")

    #   Create network
    net = make_net(input_img, is_training)
    fetches = [net["A"], net["S"]]

    # Setup session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True # Allows CPU or GPU (note: for inference, GPU is rarely necessary as the CNN is fairly small.)
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
                    print ("Not found = {}", saved_var_name)
                    return None

        #   Reminiscences from author's aging framework
        restore_vars["S/w"] = name2var["S/kernel"]
        restore_vars["S/b"] = name2var["S/bias"]

        return restore_vars

    #   Load trained CNN weights
    pretrained_path = "model/model_params.ckpt"
    common_vars = restore_map_vars(pretrained_path)
    saver = tf.train.Saver(var_list = common_vars)
    saver.restore(sess, pretrained_path)

    # Loop over all images in input directory
    for img_name in os.listdir(dir_in):
        ext = os.path.splitext(img_name)[1]
        if ext.lower() not in valid_images:
            continue

        img_path = dir_in + img_name
        print('Decomposing image {}'.format(img_path))

        #   Load image
        img = imread(img_path)
        img = CNNify(img) # Prep for CNN (e.g., add batch dimension)

        #   Inference: process image and get A and S out.
        results = sess.run([net["A"], net["S"]], {input_img:img, is_training:False})

        # Save results
        out_img_path = dir_out + img_name
        A = unCNNify(results[0]) # convert back to normal image format
        S = unCNNify(results[1]) # convert back to normal image format

        #   Save A
        A_name = os.path.splitext(out_img_path)[0] + "_A.png"
        print('Saving result {}'.format(A_name))
        imwrite(A_name, A)

        # Save S
        #   S € [0, +inf] --> rescale to [0,1]
        nS = S/np.max(S)
        S_name = os.path.splitext(out_img_path)[0] + "_S.png"
        imwrite(S_name, S)
        print('Saving result {}'.format(S_name))
