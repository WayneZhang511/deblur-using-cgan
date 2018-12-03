from glob import glob
import os
import scipy.misc
from scipy import signal
from scipy import ndimage
import numpy
import math

def psnr(img1, img2):
  mse = numpy.mean( (img1 - img2) ** 2 )
  if mse == 0:
      return 100
  PIXEL_MAX = 255.0
  return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# for concatencated 3 images
def calculate_avg_psnr(dir):
  image_paths = glob(os.path.join(dir, 'gopro_*.png'))

  sum_gt_psnr = 0
  sum_gt_ssim = 0
  sum_op_psnr = 0
  sum_op_ssim = 0

  for image_path in image_paths:
    image = scipy.misc.imread(image_path)

    input_img = image[:, :256, :]
    groud_truth = image[:, 256:512, :]
    output_img = image[:, 512:, :]

    gt_psnr = psnr(input_img, groud_truth)
    #gt_ssim = ssim(input_img, groud_truth)
    print('Ground truth: psnr %f\n' % (gt_psnr))

    op_psnr = psnr(groud_truth, output_img)
    #op_ssim = ssim(input_img, output_img)
    print('Output: psnr %f \n' % (op_psnr))

    sum_gt_psnr += gt_psnr
    #sum_gt_ssim += gt_ssim
    sum_op_psnr += op_psnr
    #sum_op_ssim += op_ssim

  size = len(image_paths)
  print('\n')
  print('#### Average ####')
  print('Ground truth: psnr %f \n' % (sum_gt_psnr / size))
  print('Output: psnr %f \n' % (sum_op_psnr / size))

calculate_avg_psnr('test')

