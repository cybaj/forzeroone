import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import datetime

from IPython import display
from dataset import DataGenerator

# tf.debugging.set_log_device_placement(True)
# print(tf.config.list_physical_devices())
# tf.config.experimental_run_functions_eagerly(True)
with tf.device('/device:GPU:0'):

    BATCH_SIZE = 4
    first_gen_map_width, first_gen_map_height = (4, 4)
    first_gen_map_ratio = 32
    first_gen_channel = 32

    datagen = DataGenerator().gen_batch
    dataset = tf.data.Dataset.from_generator(
         datagen,
         output_signature=(
             tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float64),
             tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float64)))
    
    train_dataset = dataset.batch(BATCH_SIZE)
    # train_dataset
    
    datagen = DataGenerator().gen_sample
    dataset = tf.data.Dataset.from_generator(
         datagen,
         output_signature=(
             tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float64),
             tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float64)))
    
    
    train_dataset = dataset.batch(BATCH_SIZE)
    
    def make_generator_model():
        model = tf.keras.Sequential()
        model.add(layers.Dense(first_gen_channel * first_gen_map_width * first_gen_map_height * first_gen_map_ratio, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
    
        model.add(layers.Reshape((first_gen_map_width, first_gen_map_height, first_gen_channel * first_gen_map_ratio)))
        print(model.output_shape)
        assert model.output_shape == (None, first_gen_map_width, first_gen_map_height, first_gen_channel * first_gen_map_ratio) # 주목: 배치사이즈로 None이 주어집니다.
    
        model.add(layers.Conv2DTranspose(first_gen_channel * first_gen_map_ratio / 2, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, first_gen_map_width * 2, first_gen_map_height * 2, first_gen_channel * first_gen_map_ratio / 2)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
    
        model.add(layers.Conv2DTranspose(first_gen_channel * first_gen_map_ratio / 4, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, first_gen_map_width * 4, first_gen_map_height * 4, first_gen_channel * first_gen_map_ratio / 4)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
    
        model.add(layers.Conv2DTranspose(first_gen_channel * first_gen_map_ratio / 8, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, first_gen_map_width * 8, first_gen_map_height * 8, first_gen_channel * first_gen_map_ratio / 8)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
    
        model.add(layers.Conv2DTranspose(first_gen_channel * first_gen_map_ratio / 16, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, first_gen_map_width * 16, first_gen_map_height * 16, first_gen_channel * first_gen_map_ratio / 16)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
    
        model.add(layers.Conv2DTranspose(first_gen_channel * first_gen_map_ratio / 32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, first_gen_map_width * 32, first_gen_map_height * 32, first_gen_channel * first_gen_map_ratio / 32)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, first_gen_map_width * 64, first_gen_map_height * 64, 3)
    
        return model
    
    generator = make_generator_model()
    
    def make_discriminator_model():
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(first_gen_channel / 2, (5, 5), strides=(2, 2), padding='same',
                                         input_shape=[256, 256, 3]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
    
        model.add(layers.Conv2D(first_gen_channel, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(first_gen_channel * 2, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
    
        model.add(layers.Conv2D(first_gen_channel * 4, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(first_gen_channel * 8, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Conv2D(first_gen_channel * 16, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(1))
    
        return model
    
    discriminator = make_discriminator_model()
    
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    
    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)
    
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    
    experiment_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, experiment_name, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    
    writer = tf.summary.create_file_writer(os.path.join("./mylogs", experiment_name))
    
    EPOCHS = 50
    noise_dim = 100
    num_examples_to_generate = 16
    
    # 이 시드를 시간이 지나도 재활용하겠습니다. 
    # (GIF 애니메이션에서 진전 내용을 시각화하는데 쉽기 때문입니다.) 
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    
    
    # `tf.function`이 어떻게 사용되는지 주목해 주세요.
    # 이 데코레이터는 함수를 "컴파일"합니다.
    # @tf.function(input_signature=(tf.TensorSpec(shape=[], dtype=tf.int64),tf.TensorSpec(shape=[None, 256,256,3], dtype=tf.float64)))
    def train_step(step, images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])
        ones = tf.ones([BATCH_SIZE, noise_dim])
        halves = tf.fill([BATCH_SIZE, noise_dim], 0.5)
        with writer.as_default():
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
              generated_images = generator(noise, training=True)
    
              real_output = discriminator(images, training=True)
              fake_output = discriminator(generated_images, training=True)
    
              gen_loss = generator_loss(fake_output)
              disc_loss = discriminator_loss(real_output, fake_output)
    
              if tf.math.floormod(step, tf.constant(100, dtype=tf.int64)) == tf.constant(0, dtype=tf.int64):
    #               tf.print(f'summaries {step}')
    
                  ones_images = generator(ones, training=False)
                  halves_images = generator(halves, training=False)
                  tf.summary.scalar("train_gen_loss", gen_loss, step=step)
                  tf.summary.scalar("train_disc_loss", disc_loss, step=step)
                  tf.summary.image("train_gen_images_from_noises", generated_images, step=step)
                  tf.summary.image("train_gen_images_from_ones", ones_images, step=step)
                  tf.summary.image("train_gen_images_from_halves", halves_images, step=step)
                  writer_flush = writer.flush()
            
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
        
    def generate_and_save_images(model, epoch, test_input):
      # `training`이 False로 맞춰진 것을 주목하세요.
      # 이렇게 하면 (배치정규화를 포함하여) 모든 층들이 추론 모드로 실행됩니다. 
      predictions = model(test_input, training=False)
    
      fig = plt.figure(figsize=(4,4))
    
      for i in range(predictions.shape[0]):
          plt.subplot(4, 4, i+1)
          plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
          plt.axis('off')
    
      plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
      plt.show()
        
    def train(dataset, epochs):
      total_steps = 0
      for epoch in range(epochs):
        start = time.time()
        step = tf.constant(0, dtype=tf.int64)
    
        try:
            for idx, image_batch in enumerate(dataset):
              if idx % 100 is 0 : print(f'Run step {idx}')
              # image_batch[1] : choose zero to positive 1 range float64 image
              train_step(tf.constant(idx, dtype=tf.int64), image_batch[1])
              total_steps += 1
    
            # GIF를 위한 이미지를 바로 생성합니다.
            display.clear_output(wait=True)
            generate_and_save_images(generator,
                                     epoch + 1,
                                     seed)
    
            # 15 에포크가 지날 때마다 모델을 저장합니다.
            # if (epoch + 1) % 15 == 0:
            if (epoch + 1) % 1 == 0:
              checkpoint.save(file_prefix = checkpoint_prefix)
    
            # print (' 에포크 {} 에서 걸린 시간은 {} 초 입니다'.format(epoch +1, time.time()-start))
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        except KeyboardInterrupt:
            print('Killed by Keyboard interuption')
            print(f'last epoch : {epoch}')
            print(f'last total step : {total_steps}')
            checkpoint.save(file_prefix = checkpoint_prefix)
            print(f'last checkpoint saved at {checkpoint_prefix}')
    
      # 마지막 에포크가 끝난 후 생성합니다.
      display.clear_output(wait=True)
      generate_and_save_images(generator,
                               epochs,
                               seed)
    
    train(train_dataset, EPOCHS)
