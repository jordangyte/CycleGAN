import os, PIL, re
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from utils import *
from generator import generator_fn
from discriminator import discriminator_fn
from cyclegan import CycleGan

# TPU configuration
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

REPLICAS = strategy.num_replicas_in_sync
AUTO = tf.data.experimental.AUTOTUNE

# model parameters
IMAGE_SIZE = [256, 256]
CHANNELS = 3
EPOCHS = 30
BATCH_SIZE = 8

# load data
PATH = os.path.dirname(os.path.realpath(__file__))

MONET_FILENAMES = tf.io.gfile.glob(str(PATH + '/data/monet_tfrec/*.tfrec'))
PHOTO_FILENAMES = tf.io.gfile.glob(str(PATH + '/data/photo_tfrec/*.tfrec'))

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

n_monet_samples = count_data_items(MONET_FILENAMES)
n_photo_samples = count_data_items(PHOTO_FILENAMES)

with strategy.scope():
    monet_generator = generator_fn(IMAGE_SIZE, CHANNELS) # transforms photos to Monet paintings
    photo_generator = generator_fn(IMAGE_SIZE, CHANNELS) # transforms Monet paintings to photos

    monet_discriminator = discriminator_fn(IMAGE_SIZE) # differentiates real Monet paintings and generated Monet paintings
    photo_discriminator = discriminator_fn(IMAGE_SIZE) # differentiates real photos and generated photos

# full dataset
gan_ds = get_gan_dataset(MONET_FILENAMES, PHOTO_FILENAMES, augment=augment_fn, repeat=True, shuffle=True, batch_size=BATCH_SIZE)

with strategy.scope():
    # generator optimizers
    lr_monet_gen = lambda: linear_schedule_with_warmup(tf.cast(monet_generator_optimizer.iterations, tf.float32), BATCH_SIZE, EPOCHS)
    lr_photo_gen = lambda: linear_schedule_with_warmup(tf.cast(photo_generator_optimizer.iterations, tf.float32), BATCH_SIZE, EPOCHS)
    
    monet_generator_optimizer = optimizers.Adam(learning_rate=lr_monet_gen, beta_1=0.5)
    photo_generator_optimizer = optimizers.Adam(learning_rate=lr_photo_gen, beta_1=0.5)

    # discriminator optimizers
    lr_monet_disc = lambda: linear_schedule_with_warmup(tf.cast(monet_discriminator_optimizer.iterations, tf.float32), BATCH_SIZE, EPOCHS)
    lr_photo_disc = lambda: linear_schedule_with_warmup(tf.cast(photo_discriminator_optimizer.iterations, tf.float32), BATCH_SIZE, EPOCHS)
    
    monet_discriminator_optimizer = optimizers.Adam(learning_rate=lr_monet_disc, beta_1=0.5)
    photo_discriminator_optimizer = optimizers.Adam(learning_rate=lr_photo_disc, beta_1=0.5)
    
    # Create GAN
    gan_model = CycleGan(monet_generator, photo_generator, 
                            monet_discriminator, photo_discriminator)

    gan_model.compile(m_gen_optimizer=monet_generator_optimizer,
                        p_gen_optimizer=photo_generator_optimizer,
                        m_disc_optimizer=monet_discriminator_optimizer,
                        p_disc_optimizer=photo_discriminator_optimizer,
                        gen_loss_fn=generator_loss,
                        disc_loss_fn=discriminator_loss,
                        cycle_loss_fn=calc_cycle_loss,
                        identity_loss_fn=identity_loss)
    
# training
history = gan_model.fit(gan_ds, 
                        steps_per_epoch=(max(n_monet_samples, n_photo_samples)//BATCH_SIZE),
                        epochs=EPOCHS).history

os.makedirs('/images/') # 

predict_and_save(load_dataset(PHOTO_FILENAMES).batch(1), monet_generator, '/images/')