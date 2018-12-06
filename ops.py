import tensorflow as tf
import vgg19

def leaky_relu(input_x, n_slop=0.2):
  return tf.maximum(input_x * n_slop, input_x)


def perceptual_loss(real_img, gen_img):
  batch_size = real_img.shape[0]
  vgg_net = Vgg19()
  vgg_net.build(tf.concat([real_img, gen_img], axis = 0))

  return tf.reduce_mean(tf.reduce_sum(tf.square(vgg_net.relu3_3[:batch_size] - vgg_net.relu3_3[batch_size:]), axis = 3))