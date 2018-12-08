from glob import glob
import os
import utils
from vgg19 import Vgg19
import tensorflow as tf
import numpy as np
from ops import *
import scipy.misc

class VGG_visul:
  def __init__(self, sess):
    self.sess = sess
    self.build_model()

  def build_model(self):
    self.input_A = tf.placeholder(tf.float32,
                                     [1, 256, 256, 3],
                                     name='input_A')
    self.input_B = tf.placeholder(tf.float32,
                                     [1, 256, 256, 3],
                                     name='input_B')

    vgg_net = Vgg19()
    vgg_net.build(tf.concat([self.input_A, self.input_B], axis = 0))

    # normalized feature maps
    self.sharp_feat = normalize(vgg_net.relu3_3[:1])
    self.blur_feat = normalize(vgg_net.relu3_3[1:])

  def visualize(self, base_dir):
    img_list = glob(os.path.join(base_dir, "0-*.png"))
    for img_path in img_list:
      img_name = os.path.basename(img_path)
      img = []
      img.append(utils.get_image(img_path))

      img =  np.array(img).astype(np.float32)

      sharp = img[:, :,:256,:]
      blur = img[:, :,256:,:]

      sharp_feature, blur_feature = self.sess.run([self.sharp_feat, self.blur_feat],
                                      feed_dict={self.input_A: sharp, self.input_B: blur})

      flat_sharp_feat = self.flat_features(sharp_feature)
      flat_blur_feat = self.flat_features(blur_feature)

      if not os.path.exists('feature_sharp'):
        os.makedirs('feature_sharp')
      if not os.path.exists('feature_blur'):
        os.makedirs('feature_blur')
      scipy.misc.imsave('{}/{}'.format('feature_sharp',img_name), flat_sharp_feat)
      scipy.misc.imsave('{}/{}'.format('feature_blur',img_name), flat_blur_feat)

  def flat_features(self, imgs):
    # input: batch * w * h * c
    # output: (w * sqrt(c)) * (h * sqrt(c))
    b, w, h, c = imgs.shape
    sqrt_c = int(c ** 0.5)
    out_w = w * sqrt_c
    out_h = h * sqrt_c
    output = np.zeros((out_w, out_h))

    for i in range(sqrt_c):
      for j in range(sqrt_c):
        output[i*w:(i+1)*w, j*h:(j+1)*h] = imgs[:,:,:,i*sqrt_c+j] * 255 / 2

    return np.round(output)


def main(_):
  base_dir = "/Users/wayne/Desktop"
  with tf.Session() as sess:
    model = VGG_visul(sess)
    model.visualize(base_dir)

if __name__ == '__main__':
    tf.app.run()
  
  



