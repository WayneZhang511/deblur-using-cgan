from glob import glob
import os
import scipy.misc
from scipy import signal
from scipy import ndimage
import numpy
import math
import tensorflow as tf

# for concatencated 3 images
def calculate_avg_psnr(dir):
  image_paths = glob(os.path.join(dir, 'gopro_*.png'))

  sum_ip_psnr = 0
  sum_ip_ssim = 0
  sum_ip_msssim = 0
  sum_op_psnr = 0
  sum_op_ssim = 0
  sum_op_msssim = 0

  with tf.Session():
    for image_path in image_paths:
      image = tf.image.decode_png(tf.read_file(image_path))

      input_img = image[:, :256, :]
      groud_truth = image[:, 256:512, :]
      output_img = image[:, 512:, :]

      ssim = tf.image.ssim(input_img, groud_truth, max_val=255)
      ms_ssim = tf.image.ssim_multiscale(input_img, groud_truth, max_val=255)
      psnr = tf.image.psnr(input_img, groud_truth, max_val=255)

      ip_ssim = ssim.eval()
      ip_ms_ssim = ms_ssim.eval()
      ip_psnr = psnr.eval()

      print('Input: psnr %f, ssim %f, ms_ssim %f' % (ip_psnr, ip_ssim, ip_ms_ssim))
      
      ssim = tf.image.ssim(output_img, groud_truth, max_val=255)
      ms_ssim = tf.image.ssim_multiscale(output_img, groud_truth, max_val=255)
      psnr = tf.image.psnr(output_img, groud_truth, max_val=255)

      op_ssim = ssim.eval()
      op_ms_ssim = ms_ssim.eval()
      op_psnr = psnr.eval()

      print('Output: psnr %f, ssim %f, ms_ssim %f\n' % (op_psnr, op_ssim, op_ms_ssim))

      sum_ip_psnr += ip_psnr
      sum_ip_ssim += ip_ssim
      sum_ip_msssim += ip_ms_ssim
      sum_op_psnr += op_psnr
      sum_op_ssim += op_ssim
      sum_op_msssim += op_ms_ssim

  size = len(image_paths)
  print('#### Average ####')
  print('Input: psnr %f, ssim %f, ms_ssim %f' % (sum_ip_psnr / size, sum_ip_ssim / size, sum_ip_msssim / size))
  print('Output: psnr %f, ssim %f, ms_ssim %f \n' % (sum_op_psnr / size, sum_op_ssim / size, sum_op_msssim / size))

# for seperated images in different folders
def calculate_avg_psnr_sep(input_img_dir, ground_truth_dir, output_img_dir):
  image_paths = glob(os.path.join(output_img_dir, '*.png'))

  sum_ip_psnr = 0
  sum_ip_ssim = 0
  sum_ip_msssim = 0
  sum_op_psnr = 0
  sum_op_ssim = 0
  sum_op_msssim = 0

  with tf.Session():
    for image_path in image_paths:
      image_name = os.path.basename(image_path)

      output_img = tf.image.decode_png(tf.read_file(image_path))
      input_img = tf.image.decode_png(tf.read_file(os.path.join(input_img_dir, image_name)))
      ground_truth = tf.image.decode_png(tf.read_file(os.path.join(ground_truth_dir, image_name)))

      ssim = tf.image.ssim(input_img, groud_truth, max_val=255)
      ms_ssim = tf.image.ssim_multiscale(input_img, groud_truth, max_val=255)
      psnr = tf.image.psnr(input_img, groud_truth, max_val=255)

      ip_ssim = ssim.eval()
      ip_ms_ssim = ms_ssim.eval()
      ip_psnr = psnr.eval()

      print('Input: psnr %f, ssim %f, ms_ssim %f' % (ip_psnr, ip_ssim, ip_ms_ssim))
      
      ssim = tf.image.ssim(output_img, groud_truth, max_val=255)
      ms_ssim = tf.image.ssim_multiscale(output_img, groud_truth, max_val=255)
      psnr = tf.image.psnr(output_img, groud_truth, max_val=255)

      op_ssim = ssim.eval()
      op_ms_ssim = ms_ssim.eval()
      op_psnr = psnr.eval()

      print('Output: psnr %f, ssim %f, ms_ssim %f\n' % (op_psnr, op_ssim, op_ms_ssim))

      sum_ip_psnr += ip_psnr
      sum_ip_ssim += ip_ssim
      sum_ip_msssim += ip_ms_ssim
      sum_op_psnr += op_psnr
      sum_op_ssim += op_ssim
      sum_op_msssim += op_ms_ssim

  size = len(image_paths)
  print('#### Average ####')
  print('Input: psnr %f, ssim %f, ms_ssim %f' % (sum_ip_psnr / size, sum_ip_ssim / size, sum_ip_msssim / size))
  print('Output: psnr %f, ssim %f, ms_ssim %f \n' % (sum_op_psnr / size, sum_op_ssim / size, sum_op_msssim / size))

# calculate_avg_psnr('/Users/wayne/Desktop')

calculate_avg_psnr_sep('blur_gamma_test', 'sharp_test', 'deblurred-sample_blur_test')

