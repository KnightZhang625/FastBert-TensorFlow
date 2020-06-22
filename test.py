# coding:utf-8

import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

def kl_divergence(y_true, y_pred):
  loss = y_true * np.log(y_true / (y_pred + 1e-3))
  return loss

batch_size = 2
num_classes = 2

def get_prob(probs, labels):
  flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * num_classes, [-1])
  flat_positions = labels + flat_offsets

  flat_probs = tf.reshape(probs, [-1])
  output_tensor = tf.gather(flat_probs, flat_positions)
  
  return output_tensor


a = tf.reshape(tf.constant([[0.8, 0.2], [0.3, 0.7]], dtype=tf.float32), [-1])
b = tf.reshape(tf.constant([[0.5, 0.6], [0.2, 0.3]], dtype=tf.float32), [-1])
# c = [1, 1]

# ac = get_prob(a, c)
# bc = get_prob(b, c)

loss = kl_divergence(a, b)
print(loss)