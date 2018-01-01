from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf

ckptdir = r'LT_ck\for_rect'

def test_model():
  saver = tf.train.Saver(max_to_keep=5)
  ckpt = tf.train.get_checkpoint_state(os.path.dirname(ckptdir + '/checkpoint'))
  if ckpt and ckpt.model_checkpoint_path:     #读取checkpoint，恢复训练状态
    # print ("found!")
    saver.restore(sess, ckpt.model_checkpoint_path)