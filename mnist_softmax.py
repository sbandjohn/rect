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

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rect_data
import sys

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  data = rect_data.read_data_sets(one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 100])
  W = tf.Variable(tf.zeros([100, 10]))
  b = tf.Variable(tf.zeros([10]))
  #y = tf.matmul(x, W) + b
  h = tf.nn.softmax(tf.sigmoid(tf.matmul(x,W) + b))
  y = h

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  
  #squared_error = tf.reduce_mean(tf.reduce_sum(tf.square(y - y_), reduction_indices=[1]))
   
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  
  
  #train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(squared_error)
  train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for i in range(100000):
    batch_xs, batch_ys = data.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i % 100 == 0:
      print("batch:", batch_xs, batch_ys)
      #print(sess.run(W), sess.run(b))
      train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: data.test.images,
                                      y_: data.test.labels}))

if __name__ == '__main__':
  tf.app.run(main=main, argv=sys.argv[0])
