from glob import glob
import os
import scipy.misc
from scipy import signal
from scipy import ndimage
import numpy
import math
import tensorflow as tf
import time

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
  
  file_dir = os.path.split(output_img_dir)[0]

  f = open(os.path.join(file_dir, "eval.txt"), "w")
  with tf.Session() as sess:

    print("Build model...")
    start_time = time.time()
    output_img = tf.placeholder(tf.uint8,
                                [720, 1280, 3],
                                name='output_img')
    input_img = tf.placeholder(tf.uint8,
                                [720, 1280, 3],
                                name='input_img')
    ground_truth = tf.placeholder(tf.uint8,
                                [720, 1280, 3],
                                name='ground_truth')
    ssim = tf.image.ssim(input_img, ground_truth, max_val=255)
    ms_ssim = tf.image.ssim_multiscale(input_img, ground_truth, max_val=255)
    psnr = tf.image.psnr(input_img, ground_truth, max_val=255)
    ssim1 = tf.image.ssim(output_img, ground_truth, max_val=255)
    ms_ssim1 = tf.image.ssim_multiscale(output_img, ground_truth, max_val=255)
    psnr1 = tf.image.psnr(output_img, ground_truth, max_val=255)
    print(time.time() - start_time)

    for image_path in image_paths:
      image_name = os.path.basename(image_path)
      op = scipy.misc.imread(image_path)
      ip = scipy.misc.imread(os.path.join(input_img_dir, image_name))
      gt = scipy.misc.imread(os.path.join(ground_truth_dir, image_name))

      # ssim = tf.image.ssim(input_img, ground_truth, max_val=255)
      # ms_ssim = tf.image.ssim_multiscale(input_img, ground_truth, max_val=255)
      # psnr = tf.image.psnr(input_img, ground_truth, max_val=255)
      # ssim1 = tf.image.ssim(output_img, ground_truth, max_val=255)
      # ms_ssim1 = tf.image.ssim_multiscale(output_img, ground_truth, max_val=255)
      # psnr1 = tf.image.psnr(output_img, ground_truth, max_val=255)
      print("Calculate...")
      start_time = time.time()
      ip_ssim, ip_ms_ssim, ip_psnr, op_ssim, op_ms_ssim, op_psnr = sess.run([ssim, ms_ssim, psnr, ssim1, ms_ssim1, psnr1], {input_img:ip, output_img:op, ground_truth:gt})
      print(time.time() - start_time)
      #ip_ssim = ssim.eval()
      #ip_ms_ssim = ms_ssim.eval()
      #ip_psnr = psnr.eval()
      print(image_name)
      print('Input: psnr %f, ssim %f, ms_ssim %f' % (ip_psnr, ip_ssim, ip_ms_ssim))
      
      f.write(image_name + '\n')
      f.write('Input: psnr %f, ssim %f, ms_ssim %f\n' % (ip_psnr, ip_ssim, ip_ms_ssim))
      
      #ssim = tf.image.ssim(output_img, ground_truth, max_val=255)
      #ms_ssim = tf.image.ssim_multiscale(output_img, ground_truth, max_val=255)
      #psnr = tf.image.psnr(output_img, ground_truth, max_val=255)

      #op_ssim = ssim.eval()
      #op_ms_ssim = ms_ssim.eval()
      #op_psnr = psnr.eval()

      print('Output: psnr %f, ssim %f, ms_ssim %f\n' % (op_psnr, op_ssim, op_ms_ssim))
      
      f.write('Output: psnr %f, ssim %f, ms_ssim %f\n\n' % (op_psnr, op_ssim, op_ms_ssim))
      
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
  f.write('#### Average ####\n')
  f.write('Input: psnr %f, ssim %f, ms_ssim %f \n' % (sum_ip_psnr / size, sum_ip_ssim / size, sum_ip_msssim / size))
  f.write('Output: psnr %f, ssim %f, ms_ssim %f \n' % (sum_op_psnr / size, sum_op_ssim / size, sum_op_msssim / size))
  
  f.close()
# calculate_avg_psnr('/Users/wayne/Desktop')

calculate_avg_psnr_sep('blur_gamma_test', 'sharp_test', './deblurred-sample_blur_test')

