
import os
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
from dcgan import dcgan
from preprocessing import train_test_val, get_imgs, train_test_val_images
from chest_xray import train_xray_model
from threading import Thread

def experiment(n_train, augmentor, test_set, n_gen, all_images, results_file):
    data = {}
    if all_images:
        # All images already loaded, so just partition them into train test val
        data = train_test_val_images(all_images, n_train, test_set)
        
    n_total_train_images = len(data['trn_norm_x'])+len(data['trn_pnm_x'])+len(data['val_norm_x'])+len(data['val_pnm_x'])
    if not (n_train == n_total_train_images):
        print(f"!!! Specified n_train {n_train} doen't match actual n_total_train_images {n_total_train_images}")
    
    # Generate synthetic xray images based on the training data
    all_norm_imgs = np.append(data['trn_norm_x'], data['val_norm_x'], axis=0)
    all_pnm_imgs = np.append(data['trn_pnm_x'], data['val_pnm_x'], axis=0)
    # n_gen should always be even, as all_norm_imgs == all_pnm_imgs
    if n_gen == None:
        n_gen = len(all_norm_imgs) + len(all_pnm_imgs)
    # generator_norm = augmentor(all_norm_imgs, n_gen // 2, 'normal').numpy()
    # generator_pnm = augmentor(all_pnm_imgs, n_gen // 2, 'pneumonia').numpy()
    # print(f"generator_norm.shape {generator_norm.shape} and generator_pnm.shape {generator_pnm.shape}")
    # n_images_generated = len(generator_norm) + len(generator_pnm)
    # print(f"n_images_generated: {n_images_generated}")
    
    # Train and test a model just using the small training set
    train_xray_model(data, n_gen, f'No gen just {n_train} train images', results_file)
    
    # Train and test a model using the small training set combined with the synthetic images
    # train_xray_model(data, n_gen,
    #                 #  f'{n_train} train images and {len(train_norm_files)+len(train_pnm_files)} dcgan images 10001 epochs',
    #                  f"{n_train} train images and {n_images_generated} dcgan images 10001 epochs",
    #                  results_file, gen_normal=generator_norm, gen_pnm=generator_pnm)



base_dir = '../../../../data3/xray_data_augmentation/'
results_file = f"{base_dir}/test_data/model_results6.json"
im_size = 160
all_norm_files = np.array([ base_dir+'NORMAL/'+file for file in os.listdir(base_dir+'NORMAL/')])
all_pnm_files = np.array([ base_dir+'PNEUMONIA/'+file for file in os.listdir(base_dir+'PNEUMONIA/')])[:len(all_norm_files)]
np.random.shuffle(all_norm_files)
np.random.shuffle(all_pnm_files)
all_images = {}
all_images['all_normal_images'] = get_imgs(all_norm_files, im_size)
all_images['all_pneumonia_images'] = get_imgs(all_pnm_files, im_size)

# Construct a constant test set for each experiment to use
n_test = 500
test_set = {}
# Save first n_test normal images and remove them from training pool
test_set['normal'] = all_images['all_normal_images'][:n_test//2]
all_images['all_normal_images'] = all_images['all_normal_images'][n_test//2:]
# Save first n_test pneumonia images and remove them from training pool
test_set['pneumonia'] = all_images['all_pneumonia_images'][:n_test//2]
all_images['all_pneumonia_images'] = all_images['all_pneumonia_images'][n_test//2:]

print('all_normal_images',all_images['all_normal_images'].shape)
print('all_pneumonia_images',all_images['all_pneumonia_images'].shape)
print('normal test',test_set['normal'].shape)
print('pneumonia test',test_set['pneumonia'].shape)


# Generate 500 DCGAN images
# n_gen = 5
# n_train = 2
n_train = 500
generator_norm = dcgan(all_images['all_normal_images'][:n_train], n_train, 'normal').numpy()
generator_pnm = dcgan(all_images['all_pneumonia_images'][:n_train], n_train, 'pneumonia').numpy()
# exit()

# Give me 2 more runs of 70 bc program cut out early
# for i in range(2):
#     experiment(
#         n_train=70, 
#         augmentor=dcgan, 
#         test_set=test_set, 
#         n_gen=70, 
#         all_images=all_images,
#         results_file=results_file
#     )

# for j in range(10, 101, 10): # j is number of original train image
# for asdf in range(100): # Repeat experiment
#     # print(f'Round {asdf}: {str(j)} training images')
#     experiment(
#         n_train=1000, 
#         augmentor=dcgan, 
#         test_set=test_set,
#         n_gen=0, 
#         all_images=all_images,
#         results_file=results_file
#     )

# export PATH=//usr/local/cuda/bin/${PATH:+:${PATH}}
# export LD_LIBRARY_PATH=//usr/local/cuda/lib64/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# export PATH="//usr/local/cuda/bin:$PATH"