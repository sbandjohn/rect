# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import tempfile

import rect_data

import tensorflow as tf

ckptdir = r'checkpoints/for_rect_3x3'

FLAGS = None

CHANNAL = 4

class CNN(object):
  def __init__(self, x):
    self.x_image = tf.reshape(x, [-1, 10, 10, 1])
	
    #self.W_conv1 = weight_variable([2, 2, 1, CHANNAL])
    self.W_conv1 = fixed_conv()
    self.b_conv1 = fixed_bias()
    #self.b_conv1 = bias_variable([CHANNAL])
    self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
    self.h_conv1_flat = tf.reshape(self.h_conv1, [-1, 10*10*CHANNAL])

    #self.W_fc1 = weight_variable([10*10*CHANNAL, 13])
    #self.b_fc1 = bias_variable([13])
    #self.y_conv = tf.matmul(self.h_conv1_flat, self.W_fc1) + self.b_fc1
    
    self.W_fc1 = weight_variable([10 * 10 * CHANNAL, 256])
    self.b_fc1 = bias_variable([256])
    self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv1_flat, self.W_fc1) + self.b_fc1)

    self.W_fc2 = weight_variable([256, 13])
    self.b_fc2 = bias_variable([13])
    self.y_conv = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2

def fixed_conv():
  M = [ [[1,-1], [-1,-1]], [[-1,1],[-1,-1]], [[-1,-1],[1,-1]], [[-1,-1],[-1,1]] ] 
  b = [ [ [ [ M[k][i][j] for k in range(CHANNAL)] ] for j in range(2)] for i in range(2) ]
  return tf.constant(b, dtype = tf.float32)

def fixed_bias():
  M = [ 0, 0, 0, 0 ]
  return tf.constant(M, dtype = tf.float32)

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def show_conv_res(lista):
  a=lista[0]
  print("show conv res")
  for c in range(CHANNAL):
    print("Channel", c)
    b = [ [a[i][j][c] for j in range(10) ] for i in range(10)]
    for l in b: print(l)
    print()

def main(_):
  # Import data
 
  # Create the model
  x = tf.placeholder(tf.float32, [None, 100])
  
  cnn = CNN(x)

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 13])
  
  y_conv = cnn.y_conv
  
  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  saver = tf.train.Saver(max_to_keep=5)
  

  with tf.Session() as sess:
  
    sess.run(tf.global_variables_initializer())
    print(sess.run(cnn.W_conv1))
	
    data = rect_data.read_data_sets(one_hot=True)
  
    if sys.argv[1] == "test":
      print("test only")
      ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckptdir + '/checkpoint'))
      if ckpt and ckpt.model_checkpoint_path:     #读取checkpoint，恢复训练状态
      # print ("found!")
        saver.restore(sess, ckpt.model_checkpoint_path)
      print('test accuracy %.6f' % accuracy.eval(feed_dict={
      x: data.test.images, y_: data.test.labels}))
      print(sess.run(cnn.W_conv1))
      print(sess.run(cnn.b_conv1))

    else: 
      print("train and save")
      graph_location = tempfile.mkdtemp()
      print('Saving graph to: %s' % graph_location)
      train_writer = tf.summary.FileWriter(graph_location)
      train_writer.add_graph(tf.get_default_graph())
      
      #取一个用来观察的图片  
      sample_image = data.train.images[3]
	  
      for i in range(500000):
        batch = data.train.next_batch(100)
        if i % 100 == 0:
          train_accuracy, loss = sess.run([accuracy, cross_entropy], feed_dict={
                            x: batch[0], y_: batch[1]})
          
          print(sess.run([cnn.W_conv1, cnn.b_conv1]))
          #卷积层的输出
          show_conv_res(sess.run(cnn.h_conv1, feed_dict={x:[sample_image]}))
          print('step %d, training accuracy %g loss function %g' % (i, train_accuracy, loss))
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

      print('test accuracy %.6f' % accuracy.eval(feed_dict={
      x: data.test.images, y_: data.test.labels}))

      print(sess.run(cnn.W_conv1))
      print(sess.run(cnn.b_conv1))  
	  
      saver = tf.train.Saver()
      saver.save(sess, ckptdir + '/identify-convnet', 1)    #保存模型  

if __name__ == '__main__':
  tf.app.run(main=main, argv=sys.argv)
