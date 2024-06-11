import tensorflow as tf

# import glob
# import imageio
import matplotlib.pyplot as plt
# import numpy as np
import os
# import PIL
from tensorflow.keras import layers
import time
# from preprocessing import get_imgs

# from IPython import display


def make_generator_model(img_h, img_w):
    model = tf.keras.Sequential()
    model.add(layers.Dense(10*10*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((10, 10, 256)))
    assert model.output_shape == (None, 10, 10, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 10, 10, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    # Adding in more layers for my image dimensionality
    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 20, 20, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 40, 40, 16)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2DTranspose(8, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 80, 80, 8)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(img_w, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    # assert model.output_shape == (None, 28, 28, 1)
    assert model.output_shape == (None, img_h, img_h, img_w) # 160 160 3

    return model

def make_discriminator_model(img_h, img_w):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[img_h, img_h, img_w]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def discriminator_loss(real_output, fake_output, cross_entropy):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output, cross_entropy):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

  

def dcgan(train_images, n_out, label):
    # label = 'NORMAL/'
    base_dir = '../../../../data3/xray_data_augmentation/'
    # norm_files = np.array([ f"{base_dir}{label}{file}" for file in os.listdir(base_dir+label)])[:300]  
    # all_pnm_files = np.array([ base_dir+'PNEUMONIA/'+file for file in os.listdir(base_dir+'PNEUMONIA/')])[:len(all_norm_files)]
    img_h = 160
    img_w = 3
    # train_images = get_imgs(files, img_h)
    # train_images = train_images.reshape(train_images.shape[0], img_h, img_h, img_w).astype('float32')

    print(f"DCGAN {len(train_images)} training images")

    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    BUFFER_SIZE = len(train_images) # 60000
    BATCH_SIZE = 32

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    generator = make_generator_model(img_h, img_w)
    print("DCGAN made generator")

    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)

    # plt.imshow(generated_image[0, :, :, 0], cmap='gray')

    discriminator = make_discriminator_model(img_h, img_w)
    print("DCGAN made discriminator")
    decision = discriminator(generated_image)
    # print(decision)

    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint_dir = base_dir+'models'
    checkpoint_prefix = os.path.join(checkpoint_dir, "dcgan")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)

    EPOCHS = 1001
    noise_dim = 100
    num_examples_to_generate = 16

    # You will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    
    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output, cross_entropy)
            disc_loss = discriminator_loss(real_output, fake_output, cross_entropy)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        
    def generate_and_save_images(model, epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(6, 6))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(f'genimages/{label}_image_at_epoch_{str(int(epoch))}.png')
        plt.show()
        
    def train(dataset, epochs):
        start = time.time()
        for epoch in range(epochs):
            print(f"Epoch {epoch}")
            for image_batch in dataset:
                train_step(image_batch)
            if epoch % 100 == 0:
                print(f"(Epoch {epoch})")
                generate_and_save_images(generator, epoch, seed)

            # Save the model every 15 epochs
            # if (epoch + 1) % 15 == 0:
            #     checkpoint.save(file_prefix = checkpoint_prefix)

        total_time = time.time()-start
        print (f'Time for {epochs} epochs is {total_time//60} min {total_time % 60} sec')

        # generate_and_save_images(generator, epochs, seed)

    train(train_dataset, EPOCHS)

    checkpoint.save(file_prefix = checkpoint_prefix)
    
    
    # Generate and return images
    seed = tf.random.normal([n_out, noise_dim])
    generate_and_save_images(generator, EPOCHS, seed)
    return generator(seed, training=False)
        