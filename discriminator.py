import tensorflow.keras.layers as L
from tensorflow.keras import Model
import tensorflow_addons as tfa
import tensorflow as tf
from utils import encode


def discriminator_fn(IMAGE_SIZE):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = L.Input(shape=[*IMAGE_SIZE, 3])

    x = inp

    enc1 = encode(x, 64, (4,4), (2,2), apply_instancenorm=False) 
    enc2 = encode(enc1, 128, (4,4), (2,2), apply_instancenorm=True) 
    enc3 = encode(enc2, 256, (4,4), (2,2), apply_instancenorm=True) 

    zero_pad1 = L.ZeroPadding2D()(enc3) 
    conv = L.Conv2D(512, 4, strides=1,
                    kernel_initializer=initializer,
                    use_bias=False)(zero_pad1) 

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)

    leaky_relu = L.LeakyReLU()(norm1)

    zero_pad2 = L.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = L.Conv2D(1, 4, strides=1,
                    kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    return Model(inputs=inp, outputs=last)
