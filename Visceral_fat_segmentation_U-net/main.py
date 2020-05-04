from model import *
from data import *
import os
import cv2
import logging
import skimage.io as io
import skimage.transform as trans
import args
from os.path import join,exists,basename,split
from os import makedirs,mkdir
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def main():
        # ---------------------------------------------------------------------------
        # Parse the commandline
        # ---------------------------------------------------------------------------
		parser = argparse.ArgumentParser(description='U-net Visceral fat segmentation')
		parser.add_argument('--Input_folder', type=str, default='./data/train/images', help='Input path')
		parser.add_argument('--target_folder', type=str, default='./data/train/labels', help='Output path')
		parser.add_argument('--mode', type=str, default='train', help='mode could be train or test')
		args = parser.parse_args()


		if args.mode is 'train':
			print('[i] Image directory:         ', args.Input_folder)
			print('[i] Labels directory:       ', args.target_folder)
			print('[i] Training...')
			data_gen_args = dict(width_shift_range=0.15,
		    	        height_shift_range=0.15,
		        	    shear_range=0.05,
		            	zoom_range=0.25,
		            	horizontal_flip=True)
			myGene = trainGenerator(4,args.Input_folder,args.target_folder,data_gen_args,save_to_dir ='data/train/aug_New')
			model_v = unet()

			model_checkpoint = ModelCheckpoint('unet_V_seg.hdf5', monitor='loss',verbose=1, save_best_only=True)
			model_v.fit_generator(myGene,steps_per_epoch=330,epochs=3,callbacks=[model_checkpoint])
			model_v = unet(pretrained_weights='unet_V_seg.hdf5')
			testGene = testGenerator(args.Input_folder,args.target_folder,137)
			score=model_v.evaluate_generator(testGene,137,verbose=1)
			print("Loss: ", score[0], " Binary_accuracy: ", score[1], " Mean Absolute Error: ", score[2], " Dice Coefficient: ", score[3])

		else:
			print('[i] Image directory:         ', args.Input_folder.replace('/train/images','/test/Test_results_A2B'))
			print('[i] Output directory:       ', args.target_folder.replace('/train/labels','/test/Predictions'))
			image_name_arr = glob.glob(join(args., "*.png"))
			for index,item in enumerate(image_name_arr):
				dir,name=os.path.split(item)
				img = io.imread(item, as_gray=True)
				if img.dtype == np.uint8:
					img = img / 255
				elif img.dtype == np.uint16:
					img = img / 65535
				img = np.reshape(img, img.shape + (1,))
				img = np.reshape(img, (1,) + img.shape)
				results_v = model_v.predict(img)
				img_v = np.tile((255*results_v[0,:, :, 0])[...,np.newaxis],(1,1,3))
				ret, mask_v = cv2.threshold(img_v, 100, 255, cv2.THRESH_BINARY)
				if not exists(join(args.output_dir,"Results_V")):
					makedirs(join(args.output_dir,"Results_V"))
				cv2.imwrite(join(args.output_dir,"Results_V","Resulted_"+name),mask_v)
		return 0
if __name__ == '__main__':
	sys.exit(dicom2patch())