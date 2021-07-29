from PIL import Image
import pandas as pd
import numpy as np
import pydicom, os
from tqdm import tqdm
from pydicom.pixel_data_handlers.util import apply_voi_lut

def dicom2array(path, voi_lut=True, fix_monochrome=True):
    """ Convert dicom file to numpy array 
    
    Args:
        path (str): Path to the dicom file to be converted
        voi_lut (bool): Whether or not VOI LUT is available
        fix_monochrome (bool): Whether or not to apply monochrome fix
        
    Returns:
        Numpy array of the respective dicom file 
        
    """
    # Use the pydicom library to read the dicom file
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to 
    # transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
        
    # The XRAY may look inverted
    #   - If we want to fix this we can
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    
    # Normalize the image array and return
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

train_df = pd.read_csv("./input/train.csv")
ss_df = pd.read_csv("./input/ss.csv")

save_images_train  = True
save_images_test = True
save_labels=  True
single_class = True

#Save images
if save_images_train:
	if not os.path.exists("./yolov5/covid19/images"):
		os.mkdir("./yolov5/covid19/images")
	if not os.path.exists("./yolov5/covid19/labels"):
		os.mkdir("./yolov5/covid19/labels")
	print("GENERATING TRAINING IMAGES")
	tk = tqdm(enumerate(train_df.iterrows()), total=len(train_df))
	for i, (_, row) in tk:
		arr = dicom2array(row.dcm_path)
		arr = np.stack((arr, arr, arr), axis=-1)
		im = Image.fromarray(arr)
		im.save("./yolov5/covid19/images/{}.jpg".format(os.path.split(row.dcm_path)[1][:-4]))

if save_images_test:
	if not os.path.exists("./yolov5/covid19_test_images"):
		os.mkdir("./yolov5/covid19_test_images")
	print("GENERATING TEST IMAGES")
	tk = tqdm(enumerate(ss_df.iterrows()), total=len(ss_df))
	for i, (_, row) in tk:
		if row.id.split("_")[1]=="image":
			arr = dicom2array(row.dcm_path)
			arr = np.stack((arr, arr, arr), axis=-1)
			im = Image.fromarray(arr)
			im.save("./yolov5/covid19_test_images/{}.jpg".format(os.path.split(row.dcm_path)[1][:-4]))

#Save labels
if save_labels:
	only_bb_train_df = train_df[(train_df.xmax!=1.0)]
	LABEL_LIST = ['typical', 'atypical', 'indeterminate']

	print("GENERATING IMAGE LABELS")
	tk = tqdm(enumerate(list(only_bb_train_df.id.unique())), total=len(list(only_bb_train_df.id.unique())))
	for _, img in tk:
		sub_df = only_bb_train_df[only_bb_train_df["id"]==img].reset_index(drop=True)
		h = (sub_df.ymax-sub_df.ymin)
		w = (sub_df.xmax-sub_df.xmin)
		x_c = (sub_df.xmin+w/2)/sub_df.width
		y_c = (sub_df.ymin+h/2)/sub_df.height
	    
		textfile = open("./yolov5/covid19/labels/{}.txt".format(os.path.split(sub_df.dcm_path[0])[1][:-4]), "w")
		for i in range(len(sub_df)):
			if single_class:
				lbl = "0"
			else:
				lbl = str(LABEL_LIST.index(sub_df.human_label[i]))
			textfile.write("{} {} {} {} {}\n".format(lbl, x_c[i], y_c[i], w[i]/sub_df.width[i],h[i]/sub_df.height[i]))
		textfile.close()
