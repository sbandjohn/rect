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

orick = r'ck_fixed_conv/for_rect_3x3'
newck = r'ck_fixed_fully/for_rect_3x3'

FLAGS = None

CHANNAL = 4
CONVROW = 2
CONVCOL = 2
FULLYNUM = 256
CLASS = 13
MAXSTEP = 1000
BATCHSIZE = 100

sess = tf.Session()

def global_restore_from(savepath):
  ckpt = tf.train.get_checkpoint_state(os.path.dirname(savepath + '/checkpoint'))
  assert ckpt and ckpt.model_checkpoint_path  # 读取checkpoint，恢复训练状态
  tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)

def global_save_to(savepath):
  if savepath != None:
    saver = tf.train.Saver()
    saver.save(sess, savepath + '/identify-convnet', 1)  # 保存模型


class CNN(object):
  def __init__(self):
    self.x = tf.placeholder(tf.float32, [None, 100])
    self.x_image = tf.reshape(self.x, [-1, 10, 10, 1])


  def construct_fixed_conv(self):
    #self.W_conv1 = weight_variable([2, 2, 1, CHANNAL])
    self.W_conv1 = fixed_conv()
    self.b_conv1 = fixed_bias()
    #self.b_conv1 = bias_variable([CHANNAL])
    self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
    self.h_conv1_flat = tf.reshape(self.h_conv1, [-1, 10*10*CHANNAL])

    self.W_fc1 = weight_variable([10 * 10 * CHANNAL, FULLYNUM])
    self.b_fc1 = bias_variable([FULLYNUM])
    self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv1_flat, self.W_fc1) + self.b_fc1)

    self.W_fc2 = weight_variable([FULLYNUM, CLASS])
    self.b_fc2 = bias_variable([CLASS])
    self.y_conv = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2

  def construct_fixed_fully(self, ori):
    self.W_conv1 = weight_variable([CONVROW, CONVCOL, 1, CHANNAL])
    self.b_conv1 = bias_variable([CHANNAL])
    self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
    self.h_conv1_flat = tf.reshape(self.h_conv1, [-1, 10*10*CHANNAL])

    W_fc1, b_fc1, W_fc2, b_fc2 = sess.run([ori.W_fc1, ori.b_fc1, ori.W_fc2, ori.b_fc2])

    self.W_fc1 = tf.constant(W_fc1, dtype=tf.float32)
    self.b_fc1 = tf.constant(b_fc1, dtype=tf.float32)
    self.W_fc2 = tf.constant(W_fc2, dtype=tf.float32)
    self.b_fc2 = tf.constant(b_fc2, dtype=tf.float32)

    self.h_fc1 = tf.nn.relu(tf.matmul(self.h_conv1_flat, self.W_fc1) + self.b_fc1)
    self.y_conv = tf.matmul(self.h_fc1, self.W_fc2) + self.b_fc2


  def construct_other(self):
    self.y_ = tf.placeholder(tf.float32, [None, 13])
    self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                            logits=self.y_conv)
    self.cross_entropy = tf.reduce_mean(self.cross_entropy)
    self.train_step = tf.train.AdamOptimizer(0.0001).minimize(self.cross_entropy)

    self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
    self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
    self.accuracy = tf.reduce_mean(self.correct_prediction)

  def train(self, data, savepath = None):
    print("---------------------------------------training----------------------------------")
    sess.run(tf.global_variables_initializer())

    print(sess.run(self.W_conv1))
    print("b_fc1", sess.run(self.b_fc1))
    print("b_fc2", sess.run(self.b_fc2))

     #取一个用来观察的图片
    sample_image = data.train.images[3]

    for i in range(MAXSTEP):
      batch = data.train.next_batch(BATCHSIZE)
      if i % 100 == 0:
        train_accuracy, loss = sess.run([self.accuracy, self.cross_entropy], feed_dict={
                          self.x: batch[0], self.y_: batch[1]})
        print(sess.run([self.W_conv1, self.b_conv1]))
        #卷积层的输出
        show_conv_res(sess.run(self.h_conv1, feed_dict={self.x:[sample_image]}))
        #print("b_fc1", sess.run(cnn.b_fc1))
        print("b_fc2", sess.run(self.b_fc2))
        print('step %d, training accuracy %g loss function %g' % (i, train_accuracy, loss))
      sess.run(self.train_step, feed_dict={self.x: batch[0], self.y_: batch[1]})

    self.test(data)

  def test(self, data):
    print("-------------------------testing-----------------------------------")
    print("W_conv", sess.run(self.W_conv1))
    print("b_conv1", sess.run(self.b_conv1))
    print("b_fc2", sess.run(self.b_fc2))
    print('test accuracy %.6f' % sess.run(self.accuracy, feed_dict={
      self.x: data.test.images, self.y_: data.test.labels}))

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
  # Create the model
  cnn = CNN()
  cnn.construct_fixed_conv()
  cnn.construct_other()

  global_restore_from(orick)  # get the original cnn

  cnn2 = CNN()
  cnn2.construct_fixed_fully(cnn)   # construct cnn2 from cnn
  cnn2.construct_other()

  data = rect_data.read_data_sets(one_hot = True)

  if sys.argv[1] == "test":
    print("test only")
    global_restore_from(newck)
    cnn2.test(data)
  else:
    print("train cnn2")
    cnn2.train(data)
    global_save_to(newck)

if __name__ == '__main__':
  tf.app.run(main=main, argv=sys.argv)
