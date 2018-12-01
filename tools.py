import random
import numpy as np
import os
import scipy.misc
from glob import glob

### split
def split(dir, train_frac=0.8, test_frac=0.1, sort=False):
    random.seed(0)

    files = glob(os.path.join(dir, "*.png"))
    files.sort()

    assignments = []
    assignments.extend(["train"] * int(train_frac * len(files)))
    assignments.extend(["test"] * int(test_frac * len(files)))
    assignments.extend(["val"] * int(len(files) - len(assignments)))

    if not sort:
        random.shuffle(assignments)

    for name in ["train", "val", "test"]:
        if name in assignments:
            d = os.path.join(dir, name)
            if not os.path.exists(d):
                os.makedirs(d)

    print(len(files), len(assignments))
    for inpath, assignment in zip(files, assignments):
        outpath = os.path.join(dir, assignment, os.path.basename(inpath))
        print(inpath, "->", outpath)
        os.rename(inpath, outpath)

### resize
def crop(img):
    cropx, cropy = 256, 256
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def combine(src, dataset_name):
    a_dir = os.path.join(src, 'a')
    b_dir = os.path.join(src, 'b')
    output_dir = os.path.join(src, dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for image_path in glob(os.path.join(a_dir, '*.png')):
        image_name = os.path.basename(image_path)
        
        a = scipy.misc.imread(image_path)
        b = scipy.misc.imread(os.path.join(b_dir, image_name))
        
        if a.shape[0] != 256:
            a = crop(a)
        if b.shape[0] != 256:
            b = crop(b)
            
        scipy.misc.imsave(os.path.join(output_dir, image_name), np.concatenate([a, b], axis=1))

if __name__ == '__main__':
    datast_name = 'mixed'
    ### generate dataset
    combine("/Users/wayne/Desktop/CV/project/cgan/data", datast_name)
    split("/Users/wayne/Desktop/CV/project/cgan/data" + datast_name)
