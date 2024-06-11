
import os
import time

import numpy as np
import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder 
from tensorflow.keras import layers
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense,Input,Conv2D,Flatten,MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from preprocessing import get_imgs, get_all_data
from evaluation import get_scores, save_results

# def train_xray_model(n_train):
# def train_xray_model(norm_files, pnm_files):
def train_xray_model(data, n_gen, gen_label, results_file, gen_normal=None, gen_pnm=None):

    #####################################################################################################################
    # PREPROCESSING #####################################################################################################
    #####################################################################################################################
    
    n_normal = len(data['trn_norm_x']) + len(data['val_norm_x'])
    n_pneumo = len(data['trn_pnm_x']) + len(data['val_pnm_x'])
    print(f"For training: {n_normal} normal images, {n_pneumo} pneumonia images")

    im_size = 160
    trn_norm_x = data['trn_norm_x'] # get_imgs(data['normal']['files']['train'], im_size) # [:int(len(data['normal']['files']['train']) * pcnt_train_imgs)]
    # Add generated images if they are provided
    if not (gen_normal is None):
        trn_norm_x = np.append(trn_norm_x,gen_normal,axis=0)
        data['normal']['labels']['train'] = np.append(data['normal']['labels']['train'], [data['normal']['labels']['train'][0]]*len(gen_normal))
        
    trn_pnm_x = data['trn_pnm_x'] # get_imgs(data['pneumonia']['files']['train'], im_size)
    # Add generated images if they are provided
    if not (gen_pnm is None):
        trn_pnm_x = np.append(trn_pnm_x,gen_pnm,axis=0)
        data['pneumonia']['labels']['train'] = np.append(data['pneumonia']['labels']['train'], [data['pneumonia']['labels']['train'][0]]*len(gen_pnm))
        
    tst_norm_x = data['tst_norm_x'] # get_imgs(data['normal']['files']['test'], im_size)
    tst_pnm_x = data['tst_pnm_x'] # get_imgs(data['pneumonia']['files']['test'], im_size)
    val_norm_x = data['val_norm_x'] # get_imgs(data['normal']['files']['val'], im_size)
    val_pnm_x = data['val_pnm_x'] # get_imgs(data['pneumonia']['files']['val'], im_size)
    
    n_train = len(trn_norm_x)+len(trn_pnm_x)
    n_test = len(tst_norm_x)+len(tst_pnm_x)
    n_val = len(val_norm_x)+len(val_pnm_x)

    print("train normal array shape :",trn_norm_x.shape)
    print("train pneumonia array shape :",trn_pnm_x.shape)
    print("\ntest normal array shape :",tst_norm_x.shape)
    print("test pneumonia array shape :",tst_pnm_x.shape)
    print("\nval normal array shape :",val_norm_x.shape)
    print("val pneumonia array shape :",val_pnm_x.shape)


    x_train = np.append(trn_norm_x, trn_pnm_x, axis=0)
    y_train = np.append(data['normal']['labels']['train'], data['pneumonia']['labels']['train'])
    x_test = np.append(tst_norm_x,tst_pnm_x,axis=0)
    y_test = np.append(data['normal']['labels']['test'], data['pneumonia']['labels']['test'])
    x_val = np.append(val_norm_x,val_pnm_x,axis=0)
    y_val = np.append(data['normal']['labels']['val'],data['pneumonia']['labels']['val'])

    encoder = OneHotEncoder(sparse=False) # NORMAL is [1,0] and PNEUMONIA is [0,1]
    y_train_enc= encoder.fit_transform(y_train.reshape(-1,1))
    y_test_enc= encoder.fit_transform(y_test.reshape(-1,1))
    y_val_enc= encoder.fit_transform(y_val.reshape(-1,1))


    batch_size = 16
    if n_train < batch_size:
        batch_size = n_train
    if n_test < batch_size:
        batch_size = n_test
    if n_val < batch_size:
        batch_size = n_val

    train_datagen = ImageDataGenerator(rescale = 1.0/255,
                                    samplewise_center=True,
                                    samplewise_std_normalization=True,
                                    zoom_range = 0.2, 
                                    width_shift_range=0.1,  
                                    height_shift_range=0.1)

    train_generator = train_datagen.flow(x_train,
                        y_train_enc,
                        batch_size=batch_size)


    val_datagen  = ImageDataGenerator(rescale = 1.0/255,
                                            samplewise_center=True,
                                            samplewise_std_normalization=True,
                                            zoom_range = 0.2, 
                                            width_shift_range=0.1,  
                                            height_shift_range=0.1)

    val_generator = val_datagen.flow(x_val,
                        y_val_enc,
                        batch_size=batch_size)

    test_datagen  = ImageDataGenerator(rescale = 1.0/255,
                                        samplewise_center=True,
                                        samplewise_std_normalization=True)

    test_generator = test_datagen.flow(x_test,
                        y_test_enc,
                        batch_size=batch_size)

    #####################################################################################################################
    # MODEL #############################################################################################################
    #####################################################################################################################

    n_train = len(trn_norm_x)+len(trn_pnm_x)
    epochs = 20
    learning_rate = 1e-3

    start_time = time.time()

    model = tf.keras.Sequential(name='X-ray_CNN')

    model.add(tf.keras.layers.InputLayer(input_shape=(160,160,3)))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.Dense(units=2, activation='sigmoid'))

    METRICS = ['accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')]
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=METRICS)
    hist = model.fit(train_generator,
            steps_per_epoch= x_train.shape[0] // batch_size,
            epochs= epochs,
            validation_data= val_generator,
            validation_steps= x_val.shape[0] // batch_size)
    # print(model.summary())

    end_time = time.time()
    seconds = end_time - start_time
    print(f"Training time: {seconds / 60} m {seconds % 60} s")

    #####################################################################################################################
    # EVALUATION ########################################################################################################
    #####################################################################################################################

    test_scores = get_scores(model, n_test, test_generator, 'Test set', batch_size)
    # train_scores = get_scores(model, n_train, train_generator, 'Training set', batch_size)
    # val_scores = get_scores(model, n_val, val_generator, 'Validation set', batch_size)

    save_results(results_file, {
        'n_train' : n_train, 
        'n_test': n_test, 
        'n_val': n_val,
        'n_normal': n_normal,
        'n_pneumo': n_pneumo,
        # 'train_scores': train_scores,
        'test_scores': test_scores,
        # 'val_scores': val_scores,
        'gen_label': gen_label,
        'n_generated': n_gen
    })
    
    print(f"Test accuracy: {test_scores['Accuracy']}")
