# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow as tf
import numpy as np
from . import ops

def conv1x1(input_, output_dim,
            init=tf.contrib.layers.xavier_initializer(), name='conv1x1'):
  k_h = 1
  k_w = 1
  d_h = 1
  d_w = 1
  with tf.variable_scope(name):
    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=init)
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
    return conv

def sn_conv1x1(input_, output_dim, update_collection,
              init=tf.contrib.layers.xavier_initializer(), name='sn_conv1x1'):
  with tf.variable_scope(name):
    k_h = 1
    k_w = 1
    d_h = 1
    d_w = 1
    w = tf.get_variable(
        'w', [k_h, k_w, input_.get_shape()[-1], output_dim],
        initializer=init)
    w_bar = ops.spectral_normed_weight(w, num_iters=1,
                                       update_collection=update_collection)

    conv = tf.nn.conv2d(input_, w_bar, strides=[1, d_h, d_w, 1], padding='SAME')
    return conv

def sn_non_local_block_sim(x, update_collection, name,
                           init=tf.contrib.layers.xavier_initializer()):
  with tf.variable_scope(name):
    batch_size, h, w, num_channels = x.get_shape().as_list()
    location_num = h * w
    # downsampled_num = location_num // 4
    downsampled_num = -1  # this sets the matrix dimensions automatically using
    # batch_size and num_channels

    # theta path (JG: 'theta' is the result of g(x) in Fig. 2 of SAGAN paper)
    theta = sn_conv1x1(x, num_channels // 8, update_collection, init, 'sn_conv_theta')
    theta = tf.reshape(
        theta, [batch_size, location_num, num_channels // 8])

    # phi path (JG: 'phi' is the result of f(x) in Fig. 2 of SAGAN paper)
    phi = sn_conv1x1(x, num_channels // 8, update_collection, init, 'sn_conv_phi')
    phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2, padding='SAME')
    # JG: for FOV size of [49, 49], phi will have shape [4, 25, 25, 4] at this step.
    phi = tf.reshape(
        phi, [batch_size, downsampled_num, num_channels // 8])


    attn = tf.matmul(theta, phi, transpose_b=True)
    attn = tf.nn.softmax(attn)
    print(tf.reduce_sum(attn, axis=-1))

    # g path (JG: Confusingly, 'g' is actually the result of h(x) in Fig. 2)
    g = sn_conv1x1(x, num_channels // 2, update_collection, init, 'sn_conv_g')
    g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2, padding='SAME')
    g = tf.reshape(
      g, [batch_size, downsampled_num, num_channels // 2])

    attn_g = tf.matmul(attn, g)
    attn_g = tf.reshape(attn_g, [batch_size, h, w, num_channels // 2])
    sigma = tf.get_variable(
        'sigma_ratio', [], initializer=tf.constant_initializer(0.0))
    attn_g = sn_conv1x1(attn_g, num_channels, update_collection, init, 'sn_conv_attn')
    return x + sigma * attn_g