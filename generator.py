import tensorflow.keras.layers as L
from tensorflow.keras import Model
import tensorflow as tf
from utils import encode, decode, residual_block

def generator_fn(IMAGE_SIZE, OUTPUT_CHANNELS, n_residuals_blocks=9):
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = L.Input(shape=[*IMAGE_SIZE, 3])

    enc_1 = encode(inputs, 64, (7,7), (1,1), apply_instancenorm=False)
    enc_2 = encode(enc_1, 128, (3,3), (2,2), apply_instancenorm=True)
    enc_3 = encode(enc_2, 256, (3,3), (2,2), apply_instancenorm=True)
    
    res = enc_3
    for i in range(n_residuals_blocks):
        res = residual_block(res, 256, (3,3), (1,1))
    
    x_skip = L.Concatenate()([res, enc_3])
    
    dec_1 = decode(x_skip, 128, (3,3), (2,2))
    x_skip = L.Concatenate()([dec_1, enc_2])
    
    dec_2 = decode(x_skip, 64, (3,3), (2,2))
    x_skip = L.Concatenate()([dec_2, enc_1])
    
    last = L.Conv2D(OUTPUT_CHANNELS, (7,7), 
                    strides=(1,1), padding='same', 
                    kernel_initializer=initializer, 
                    use_bias=False, 
                    activation='tanh')(x_skip)
    
    return Model(inputs=inputs, outputs=last)
