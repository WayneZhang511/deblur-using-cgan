import os
from glob import glob
import shutil

base_dir = '/home/yz3243/GOPRO_Large/train'
folders = os.listdir(base_dir)

a_dir = 'a'
if not os.path.exists(a_dir):
  os.makedirs(a_dir)
b_dir = 'b'
if not os.path.exists(b_dir):
  os.makedirs(b_dir)

counter = 1
for folder in folders:
  path = os.path.join(base_dir, folder)
  sharp_path = os.path.join(path, 'sharp')
  blur_path = os.path.join(path, 'blur_gamma')
  img_a_dirs = glob(os.path.join(sharp_path, '*.png'))

  for img_a_dir in img_a_dirs:
    img_name = os.path.basename(img_a_dir)
    img_b_dir = os.path.join(blur_path, img_name)

    new_img_name = str(counter) + '.png'
    shutil.copyfile(img_a_dir, os.path.join(a_dir, new_img_name))
    shutil.copyfile(img_b_dir, os.path.join(b_dir, new_img_name))
    counter += 1



