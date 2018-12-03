import tensorflow as tf
from glob import glob

def compute_psnr(img1, img2):
  psnr = tf.image.psnr(img1, img2, max_val=255)

  return psnr

def compute_ssim(img1, img2):
  ssim = tf.image.ssim(img1, img2, max_val=255)

  return ssim

# for concatencated 3 images
def calculate_avg_psnr(dir):
  image_paths = glob(os.path.join(dir, '*.png'))

  sum_gt_psnr = 0
  sum_gt_ssim = 0
  sum_op_psnr = 0
  sum_op_ssim = 0

  for image_path in image_paths:
    image = tf.decode_png(image_path)

    input_img = image[:, :256, :]
    groud_truth = image[:, 256:512, :]
    output_img = image[:, 512:, :]

    gt_psnr = compute_psnr(input_img, groud_truth)
    gt_ssim = compute_ssim(input_img, groud_truth)
    print('Ground truth: psnr %f, ssim %f \n' % (gt_psnr[0, 0], gt_ssim[0, 0]))

    op_psnr = compute_psnr(input_img, output_img)
    op_ssim = compute_ssim(input_img, output_img)
    print('Output: psnr %f, ssim %f \n' % (op_psnr[0, 0], op_ssim[0, 0]))

    sum_gt_psnr += gt_psnr[0, 0]
    sum_gt_ssim += gt_ssim[0, 0]
    sum_op_psnr += op_psnr[0, 0]
    sum_op_ssim += op_ssim[0, 0]

  size = len(image_paths)
  print('\n')
  print('#### Average ####')
  print('Ground truth: psnr %f, ssim %f \n' % (sum_gt_psnr / size, sum_gt_ssim / size))
  print('Output: psnr %f, ssim %f \n' % (sum_op_psnr / size, sum_op_ssim / size))



