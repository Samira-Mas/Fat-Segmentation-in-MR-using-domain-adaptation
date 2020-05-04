from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import cv2
import glob
import skimage.io as io
import skimage.transform as trans
# from skimage.filters import threshold_multiotsu


def normalize(arr, N=254, eps=1e-14):
    """
    TO normalize an image by mapping its [Min,Max] into the interval [0,255]
    :param arr: Input (2D or 3D) array of image
    :param N: Scaling factor
    :param eps:
    :return: Normalized Image
    """
    arr = arr.astype(np.float64)
    output = N*(arr-np.min(arr))/(np.max(arr)-np.min(arr)+eps)
    return output
def adjustData(img,mask,flag_multi_class,num_class):
    if(np.max(img) > 1):
        if img.dtype == np.uint8:
            img = img / 255
        elif img.dtype == np.uint16:
            img = img / 65535
        if mask.dtype == np.uint8:
            mask = mask / 255
        elif img.dtype == np.uint16:
           mask = mask / 65535
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 1,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)



def testGenerator(test_path,test_path2,num_image = 60,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%(i)),as_gray = as_gray)
        mask=io.imread(os.path.join(test_path2,"%d.png"%(i)),as_gray = as_gray)
        if img.dtype == np.uint8:
            img = img / 255
        elif img.dtype == np.uint16:
            img = img / 65535
        if mask.dtype == np.uint8:
            mask = mask / 255
        elif img.dtype == np.uint16:
           mask = mask / 65535
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        mask = trans.resize(mask, target_size)
        mask = np.reshape(mask, mask.shape + (1,)) if (not flag_multi_class) else img
        mask = np.reshape(mask, (1,) + mask.shape)
        yield img,mask

