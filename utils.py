import scipy.misc
import numpy as np

def get_image(file_name):
    image = scipy.misc.imread(file_name)
    # normalization
    image = image / 127.5 - 1
    
    return image


# def save_images(images, size, file_name):
#     return scipy.misc.imsave(file_name, merge_images(images, size))

def merge_images1(images, size):
    shape = images.shape
    h,w,ch = shape[1], shape[2], shape[3]
    imgs = np.zeros([h * size[0], w * size[1], ch])
    print(imgs.shape)
    print(images.shape)
    for idx, image in enumerate(images):
        print(image.shape)
        i = idx % size[1]
        j = idx // size[0]
        imgs[j * h: j * h + h, i * w: i * w + w, :] = image

    # deal with grey scale image
    if imgs.shape[2] == 1:
        return imgs[:,:,0]
    else:
        return imgs

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.    
