
import numpy as np
from PIL import Image
import os

def val_split(train_data, pcnt):
    i = int(len(train_data) * pcnt)
    return train_data[:i], train_data[i:] # train, val

def train_test_val_images(all_images, n_train, test_set, val_ratio=0.9):
    data = {}
    data['normal'] = {} # = train_test_val(train_norm_files, test_norm_files, 0.9, 'normal')
    data['pneumonia'] = {} # train_test_val(train_pnm_files, test_pnm_files, 0.9, 'pneumonia')
    train_val_norm = all_images['all_normal_images'][:n_train//2]
    data['trn_norm_x'], data['val_norm_x'] = val_split(train_val_norm, val_ratio)
    data['tst_norm_x'] = test_set['normal'] # all_images['all_normal_images'][n_train//2:]
    
    train_val_pnm = all_images['all_pneumonia_images'][:n_train//2]
    data['trn_pnm_x'], data['val_pnm_x'] = val_split(train_val_pnm, val_ratio)
    data['tst_pnm_x'] = test_set['pneumonia'] # all_images['all_pneumonia_images'][n_train//2:]
    
    data['normal']['labels'] = {
        'train': np.array(['normal']*len(data['trn_norm_x'])),
        'test': np.array(['normal']*len(data['tst_norm_x'])),
        'val': np.array(['normal']*len(data['val_norm_x']))
    }
    data['pneumonia']['labels'] = {
        'train': np.array(['pneumonia']*len(data['trn_pnm_x'])),
        'test': np.array(['pneumonia']*len(data['tst_pnm_x'])),
        'val': np.array(['pneumonia']*len(data['val_pnm_x']))
    }
    
    return data
    

def train_test_val(train_data, test_data, val_ratio, label):
    train, val = val_split(train_data, val_ratio)
    return {
        'files': {
            'train': train,
            'test': test_data,
            'val': val
        },
        'labels': {
            'train': np.array([label]*len(train)),
            'test': np.array([label]*len(test_data)),
            'val': np.array([label]*len(val))
        }
    }

def train_test_val_old(data, ratios, label, n_train):
    i1 = int(len(data) * ratios['train'])
    i2 = int(len(data) * ratios['traintest'])
    train = data[:i1]
    test = data[i1:i2]
    val = data[i2:]
    
    # extra_train = get_gen_images()
    all_train = np.concatenate((train[:n_train], extra_train))
    # n_train = int(len(train) * pcnt_train_imgs)
    return { 
        'files': {
            'train': all_train, 
            'test': test, 
            'val': val
        },
        'labels': {
            'train': np.array([label]*len(all_train)),
            'test': np.array([label]*len(test)),
            'val': np.array([label]*len(val))
        }
    }
    
def get_imgs(files, im_size):
    print(files[0],'...')
    images = np.zeros((len(files), im_size, im_size, 3), dtype='float32')
    for i, img_file in enumerate(files):
        img = np.array(Image.open(img_file).resize((im_size, im_size)).convert("RGB"))
        images[i] = img
        # print(i, end=' ')
    # print()
    return images

def get_all_data(n_train):
    # # With % train / test / val 80 / 10 / 10 taken from same directory
    base_dir = '../../../../data3/xray_data_augmentation/'
    all_norm_files = np.array([ base_dir+'NORMAL/'+file for file in os.listdir(base_dir+'NORMAL/')])
    all_pnm_files = np.array([ base_dir+'PNEUMONIA/'+file for file in os.listdir(base_dir+'PNEUMONIA/')])[:len(all_norm_files)]
    np.random.shuffle(all_norm_files)
    np.random.shuffle(all_pnm_files)
    print(f"Total {len(all_norm_files)} normal, {len(all_pnm_files)} pneumonia images")

    # n_normal = len(all_norm_files)
    # n_pneumo = n_normal # len(all_pnm_files)
    # all_norm_files = all_norm_files[:n_normal]
    # all_pnm_files = all_pnm_files[:n_pneumo]

    data = {}
    data['normal'] = train_test_val(all_norm_files, {'train': 0.8, 'traintest': 0.9}, 'normal', n_train)
    data['pneumonia'] = train_test_val(all_pnm_files, {'train': 0.8, 'traintest': 0.9}, 'pneumonia', n_train)
    return data