import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Tensorflow
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow_examples.models.pix2pix import pix2pix

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

BUFFER_SIZE = 1000
BATCH_SIZE = 1

IMG_WIDTH = 256
IMG_HEIGHT = 256

AUTOTUNE = tf.data.AUTOTUNE

def input_pipeline():
    # Input
    dataset, metadata = tfds.load('cycle_gan/horse2zebra', with_info=True, as_supervised=True)

    train_horses, train_zebras = dataset['trainA'], dataset['trainB']
    test_horses, test_zebras = dataset['testA'], dataset['testB']

    ''' Preproccessing '''
    # Random Jittering
    def random_crop(image):
        image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

        return image
    
    def random_mirror(image):
        image = tf.image.random_flip_left_right(image)

        return image

    def random_jitter(image):
        # Resizing to 286 x 286 x 3
        image = tf.image.resize(image, [286, 286],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Randomly cropping to 256 x 256 x 3
        image = random_crop(image)

        # Random mirroring
        image = random_mirror(image)

        return image

    # Normalizing the images to [-1, 1]
    def normalize(image):
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        return image
    
    # Make preprocess methods
    def preprocess_image_train(image, label):
        image = random_jitter(image)
        image = normalize(image)

        return image

    def preprocess_image_test(image, label):
        image = normalize(image)

        return image
    
    train_horses = train_horses.map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    train_zebras = train_zebras.map(
        preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    test_horses = test_horses.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

    test_zebras = test_zebras.map(
        preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE)

def main():
    input_pipeline()
    
if __name__ is '__main__':
    main()