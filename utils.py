import scipy.misc
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as euc_dist

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

def inv_transform(images):
    return np.round((images + 1) * 255 / 2)

def generate_patches(image):
    m,n,_ = image.shape

    patches = []

    row = 0
    i_stride = 232
    for i in range(3):
        col = 0
        j_stride = 205
        for j in range(6):
            if j == 5:
                col -= 1

            patches.append(image[row:row+256, col:col+256, :])
            col += j_stride

        row += i_stride

    return patches


def merge_images(canvas, imgs):
    counter = 0

    row = 0
    i_stride = 232
    for i in range(3):
        col = 0
        j_stride = 205
        for j in range(6):
            if j == 5:
                col -= 1

            canvas_mask = binary_mask(canvas)
            tmp_img = np.zeros_like(canvas)
            tmp_img[row:row+256, col:col+256, :] = imgs[counter]
            tmp_img_mask = binary_mask(tmp_img)

            canvas = blend_image_pair(canvas, canvas_mask, tmp_img, tmp_img_mask)
            counter += 1
            col += j_stride

        row += i_stride

    return canvas


def binary_mask(img):
    # Input:
    #     img: source image, shape (m, n, 3)
    # Output:
    #     mask: image of shape (m, n) and type 'int'. For pixel [i, j] of mask, if img[i, j] > 0 
    #           in any of its channels, mask[i, j] = 1. Else, (if img[i, j] = 0), mask[i, j] = 0.
    
    mask = np.zeros((img.shape[0], img.shape[1]))
    for i in range(3):
        mask[np.where(img[:,:,i] != 0)] = 1
    
    return mask



def blend_image_pair(src_img, src_mask, dst_img, dst_mask):
    # Given two images and their binary masks, the two images are blended.
    # 
    # Input:
    #     src_img: First image to be blended, shape (m, n, 3)
    #     src_mask: src_img's binary mask, shape (m, n)
    #     dst_img: Second image to be blended, shape (m, n, 3)
    #     dst_mask: dst_img's binary mask, shape (m, n)
    #     mode: Blending mode, either "overlay" or "blend"
    # Output:
    #     Blended image of shape (m, n, 3)

    w, d, _ = src_img.shape
    blend_img = np.zeros(src_img.shape)
    
    src_dist = euc_dist(src_mask)
    dst_dist = euc_dist(dst_mask)
    for i in range(w):
        for j in range(d):
            if src_mask[i][j] == 1:
                blend_img[i,j,:] = src_img[i,j,:]
            if dst_mask[i][j] == 1:
                blend_img[i,j,:] = dst_img[i,j,:]
            if src_mask[i][j] == 1 and dst_mask[i][j] == 1:
                w1 = src_dist[i,j]
                w2 = dst_dist[i,j]
                blend_img[i,j,:] = (w1 * src_img[i,j,:] + w2 * dst_img[i,j,:]) / (w1 + w2)
                    
    return blend_img.astype(np.uint8)





