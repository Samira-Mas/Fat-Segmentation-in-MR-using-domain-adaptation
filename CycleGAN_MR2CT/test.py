import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tf2lib as tl
import cv2
import data
import module

import os
def normalize(arr,eps=0.000001):
    return 2*((arr-np.min(arr))/(np.max(arr)-np.min(arr)+eps))-1
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# ==============================================================================
# =                                   param                                    =
# ==============================================================================
py.arg('--dataset', default='MRI2CT')
py.arg('--datasets_dir', default='/home/mip/Downloads/Cyc_MRI2CT/datasets')
py.arg('--experiment_dir')
py.arg('--batch_size', type=int, default=1)
test_args = py.args()
test_args.experiment_dir='/home/mip/Downloads/Cyc_MRI2CT/output/MRI2CT'
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)


# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# data
A_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'MRI_Bagci'), '*.png')
# print(len(A_img_paths_test))
# B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'CT_test'), '*.png')
A_dataset_test = data.make_dataset(A_img_paths_test, args.batch_size, args.load_size, args.crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)
# B_dataset_test = data.make_dataset(B_img_paths_test, args.batch_size, args.load_size, args.crop_size,
#                                     training=False, drop_remainder=False, shuffle=False, repeat=1)

# model
G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))

# resotre
tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join('/home/mip/Downloads/Cyc_MRI2CT/output2/MRI2CT2', 'ckpt')).restore()

def CLIP(arr,WL=.4,WW=.7):

    if (WL-WW)>-1:
        arr[arr<(WL-WW)]=(WL-WW)
    if (WL - WW) < 1:
        arr[arr>(WL+WW)]= WL+WW
    return arr
@tf.function
def sample_A2B(A):
    A2B = G_A2B(A, training=False)
    # A2B2A = G_B2A(A2B, training=False)
    return A2B#, A2B2A


# @tf.function
# def sample_B2A(B):
#     B2A = G_B2A(B, training=False)
#     # B2A2B = G_A2B(B2A, training=False)
#     return B2A


# run
save_dir = py.join(args.experiment_dir, 'samples_testing_Bagci_v3', 'A2B')
py.mkdir(save_dir)
i = 0
for A in A_dataset_test:
    A2B= sample_A2B(A)
    # B2A = sample_B2A(B)
    for A_i, A2B_i in zip(A, A2B):
        # print(np.min(A_i.numpy()),np.max(A_i.numpy()))
        # AA1=CLIP(,.4,.75)
        # AA2 = 0#normalize(CLIP(A2B_i.numpy()[np.newaxis, ...],.35,.65))
        #
        AA=127.0*(A2B_i.numpy()+1.0)
        # print(np.min(AA), np.max(AA))
        # print(A2B_i.shape)
        img = im.immerge(np.concatenate([A_i.numpy()[np.newaxis,...],A2B_i.numpy()[np.newaxis,...]], axis=0), n_rows=1)
        # im.imwrite(img, py.join(save_dir, py.name_ext(A_img_paths_test[i].replace('.png','.jpg'))))
        cv2.imwrite(py.join(save_dir, py.name_ext(A_img_paths_test[i])),AA)
        i += 1
#
# save_dir = py.join(args.experiment_dir, 'samples_testing', 'B2A')
# py.mkdir(save_dir)
# i = 0
# for B in B_dataset_test:
#     B2A, B2A2B = sample_B2A(B)
#     for B_i, B2A_i, B2A2B_i in zip(B, B2A, B2A2B):
#         img = np.concatenate([B_i.numpy(), B2A_i.numpy(), B2A2B_i.numpy()], axis=1)
#         cv2.imwrite( py.join(save_dir, py.name_ext(B_img_paths_test[i])),normalize(img))
#         i += 1
