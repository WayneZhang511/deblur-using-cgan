import tensorflow as tf
from vgg19 import Vgg19
from utils import transform

def leaky_relu(input_x, n_slop=0.2):
  return tf.maximum(input_x * n_slop, input_x)


def perceptual_loss(real_img, gen_img):
  batch_size = real_img.shape[0]
  vgg_net = Vgg19()
  vgg_net.build(tf.concat([real_img, gen_img], axis = 0))

  return tf.reduce_mean(tf.reduce_sum(tf.square(normalize(vgg_net.relu3_3[:batch_size]) - normalize(vgg_net.relu3_3[batch_size:])), axis = 3))

def normalize(tensor):
  tensor = tf.div(
   tf.subtract(
      tensor, 
      tf.reduce_min(tensor)
   ), 
   tf.subtract(
      tf.reduce_max(tensor), 
      tf.reduce_min(tensor)
   ) / 2.0
  )

  return tensor
