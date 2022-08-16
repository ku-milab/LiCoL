import tensorflow.keras as keras
import tensorflow as tf
tf.random.Generator = None
import tensorflow_addons as tfa
import CMG_config as config

l = tf.keras.layers
W, H, D, c = 96, 114, 96, config.classes

def input_layer(input_shape=(W, H, D, 1), name=None):
    return l.Input(shape=input_shape)

def resc1(input_shape=(48, 57, 48, c), name=None):
    return l.Input(shape=input_shape)
def resc2(input_shape=(24, 29, 24, c), name=None):
    return l.Input(shape=input_shape)
def resc3(input_shape=(12, 15, 12, c), name=None):
    return l.Input(shape=input_shape)
def resc4(input_shape=(6, 8, 6, c), name=None):
    return l.Input(shape=input_shape)
def resc5(input_shape=(3, 4, 3, c), name=None):
    return l.Input(shape=input_shape)

def conv(f, k=3, s=1, p='same', act=None, b=True, k_init="he_normal", dilation_rate=(1,1,1), out_p=None, k_reg=0.00001, k_const=None, rank=3, name=None):
    if rank==3:
        conv_layer = l.Conv3D
    elif rank==2:
        conv_layer = l.Conv2D
    elif rank==1:
        conv_layer = l.Conv1D
    else:
        raise Exception("Conv Rank Error!!")

    if k_reg:
        k_reg = tf.keras.regularizers.l2(k_reg)
    if k_const:
        k_const = tf.keras.constraints.MaxNorm(max_value=k_const, axis=[0, 1, 2, 3])

    return conv_layer(filters=f, kernel_size=k, strides=s, padding=p, activation=act, use_bias=b, kernel_initializer=k_init,
                      dilation_rate=dilation_rate, bias_initializer='zeros', kernel_regularizer=k_reg, kernel_constraint=k_const, name=name)

def conv_transpose(f, k=2, s=3, p='valid', out_p='auto', act=None, b=True, dilation_rate=(1,1,1), k_init="he_normal", k_reg=0.00001,
                   k_const=None, rank=3, name=None):
    if rank==3:
        conv_transpose_layer = l.Conv3DTranspose
    elif rank==2:
        conv_transpose_layer = l.Conv2DTranspose
    else:
        raise Exception("Conv Transpose Rank Error!!")

    if k_reg:
        k_reg = tf.keras.regularizers.l2(k_reg)
    if k_const:
        k_const = tf.keras.constraints.MaxNorm(max_value=k_const, axis=[0, 1, 2, 3])

    if out_p=="auto":
        out_p = (W%s, H%s, D%s)

    return conv_transpose_layer(filters=f, kernel_size=k, strides=s, padding=p, output_padding=out_p,
                                activation=act, use_bias=b, kernel_initializer=k_init, bias_initializer='zeros',
                                dilation_rate=dilation_rate, kernel_regularizer=k_reg, kernel_constraint=k_const, name=name)

def batch_norm(m=0.99, e=1e-3, name=None):
    return l.BatchNormalization(momentum=m, epsilon=e)

def relu(name=None):
    return l.ReLU()

def tanh(x, name=None):
    return keras.activations.tanh(x)

def linear(x, name=None):
    return keras.activations.linear(x)

def sigmoid(x, name=None):
    return keras.activations.sigmoid(x)

def identity(x, name=None):
    return tf.identity(x)

def leaky_relu(name=None):
    return l.LeakyReLU(alpha=0.2)

def concat(axis=-1, name=None):
    return l.Concatenate(axis=axis)

def dropout(rate=0.5, name=None):
    return l.Dropout(rate=rate)

def flatten(name=None):
    return l.Flatten()

def cropping(size, mode="edge", name=None):
    if len(size)==3:
        crop_layer = l.Cropping3D
    elif len(size)==2:
        crop_layer = l.Cropping2D
    elif len(size)==1:
        crop_layer = l.Cropping1D
    else:
        raise Exception("Crop Rank Error!!")

    if mode=="edge":
        cropping=[[s//2, s-s//2] for s in size]
    elif mode=="front":
        cropping=[[s,0] for s in size]
    elif mode=="end":
        cropping=[[0,s] for s in size]
    else:
        raise Exception("Crop Mode Error!!")

    return crop_layer(cropping=cropping)

def zero_padding(pad, rank=3, name=None):
    if rank==3:
        zero_pad = l.ZeroPadding3D
    elif rank==2:
        zero_pad = l.ZeroPadding2D
    elif rank==1:
        zero_pad = l.ZeroPadding1D
    else:
        raise Exception("Zero_padding Rank Error!!")
    return zero_pad(padding=pad)

def maxpool(k=2, s=2, p="valid", rank=3, name=None):
    if rank==3:
        maxpool_layer = l.MaxPooling3D
    elif rank==2:
        maxpool_layer = l.MaxPooling2D
    elif rank==1:
        maxpool_layer = l.MaxPooling1D
    else:
        raise Exception("Maxpool Rank Error!!")
    return maxpool_layer(pool_size=k, strides=s, padding=p)

def avgpool(k, s, p="same", rank=3, name=None):
    if rank==3:
        avgpool_layer = l.AveragePooling3D
    elif rank==2:
        avgpool_layer = l.AveragePooling2D
    elif rank==1:
        avgpool_layer = l.AveragePooling1D
    else:
        raise Exception("Avgpool Rank Error!!")

    return avgpool_layer(pool_size=k, strides=s, padding=p)

def global_avgpool(rank=3, name=None):
    if rank==3:
        global_avgpool_layer = l.GlobalAveragePooling3D
    elif rank==2:
        global_avgpool_layer = l.GlobalAveragePooling2D
    elif rank==1:
        global_avgpool_layer = l.GlobalAveragePooling1D
    else:
        raise Exception("global_avgpool Rank Error!!")

    return global_avgpool_layer()

def upsample(rank=3, s=2, name=None):
    if rank==3:
        upsampling = l.UpSampling3D
    elif rank==2:
        upsampling = l.UpSampling2D
    elif rank==1:
        upsampling = l.UpSampling1D
    else:
        raise Exception("Upsample Rank Error!!")
    return upsampling(size=s)

def dense(f, act=None, b=True, k_init="he_normal", k_reg=0.00001, k_const=None, name=None):
    if k_reg:
        k_reg = tf.keras.regularizers.l2(k_reg)
    if k_const:
        k_const=tf.keras.constraints.MaxNorm(max_value=k_const, axis=[0])

    return l.Dense(units=f, activation=act, use_bias=b, kernel_initializer=k_init,
                   bias_initializer="zeros", kernel_regularizer=k_reg, kernel_constraint=k_const)

def softmax(x, name=None):
    return l.Softmax()(x)