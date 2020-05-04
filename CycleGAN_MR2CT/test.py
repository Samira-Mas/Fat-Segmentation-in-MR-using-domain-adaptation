import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import Utils as utils
import cv2
import data
import module
from os.path import join,exists,basename
from os import makedirs,mkdir
import os
def normalize(arr,eps=0.000001):
    return 2*((arr-np.min(arr))/(np.max(arr)-np.min(arr)+eps))-1
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# ==============================================================================
# =                                   param                                    =
# ==============================================================================
py.arg('--datasets_dir', default='./datasets')
py.arg('--experiment_dir',default='./Results')
py.arg('--batch_size', type=int, default=1)
test_args = py.args()
args = py.args_from_yaml(join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)


# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# data
A_img_paths_test = glob.glob(join(args.datasets_dir, 'MRI_test', '*.png'))
# print(len(A_img_paths_test))
# B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'CT_test'), '*.png')
A_dataset_test = data.make_dataset(A_img_paths_test, args.batch_size, args.size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)
G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))

# resotre
utils.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), join(test_args.experiment_dir, 'checkpoints')).restore()

@tf.function
def sample_A2B(A):
    A2B = G_A2B(A, training=False)
    return A2B

save_dir = join(args.experiment_dir, 'Test_results_A2B')
if not exists(save_dir):
        makedirs(save_dir)
i = 0
for A in A_dataset_test:
    A2B= sample_A2B(A)
    for A_i, A2B_i in zip(A, A2B):
        cv2.imwrite(join(save_dir, basename(A_img_paths_test[i])),127.0*(A2B_i.numpy()+1.0))
        i += 1
