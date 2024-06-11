
# from preprocessing import train_test_val, get_imgs, train_test_val_images
# from PIL import Image
# from preprocessing import get_imgs
# from chest_xray import train_xray_model
# import numpy as np
import os
# import keras_cv


base_dir = '../../../../data3/xray_data_augmentation/'
# ckpt_path = base_dir+"finetuned_stable_diffusion.h5"

sum = 0
im_dirs = ['PNEUMONIA', 'NORMAL']
for d in im_dirs:
    n = os.listdir(base_dir+d)
    print(len(n))
    sum += len(n)
    
print(f"sum: {sum}")