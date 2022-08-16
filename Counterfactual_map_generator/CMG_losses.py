import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
K = keras.backend

def CE_loss(lbl, pred):
    return K.mean(keras.losses.categorical_crossentropy(lbl, pred))

def BCE_loss(lbl, pred):
    return K.mean(tf.nn.sigmoid_cross_entropy_with_logits(lbl, pred))

def MSE_loss(lbl, pred):
    return K.mean(keras.losses.mse(lbl, pred))

def one_sided_label_smoothing(labels):
    return np.where(labels == 1., .9, labels)

def L1_norm(effect_map):
    return tf.reduce_mean(tf.abs(effect_map))

def L2_norm(effect_map):
    return tf.sqrt(tf.reduce_mean(tf.square(effect_map)))

def discriminator_LS_loss(real, fake):
    real_loss = MSE_loss(tf.ones_like(real), real)
    fake_loss = MSE_loss(tf.zeros_like(fake), fake)
    return real_loss + fake_loss

def discriminator_loss_BCE(real, fake):
    real_loss = BCE_loss(tf.ones_like(real), real)
    fake_loss = BCE_loss(tf.zeros_like(fake), fake)
    return real_loss + fake_loss

def generator_LS_loss(target_c, fake):
    y_gen = np.expand_dims(np.ones(target_c.shape[0]).astype("float32"), axis=-1)
    return MSE_loss(y_gen, fake)

def generator_loss_BCE(target_c, fake):
    y_gen = np.expand_dims(np.ones(target_c.shape[0]).astype("float32"), axis=-1)
    return BCE_loss(y_gen, fake)

def cycle_loss(input, like_input):
    return tf.dtypes.cast(tf.reduce_mean(tf.abs(like_input - input)), tf.float32)

def tv_loss(image):
    diff_x = image[:, 1:, :, :, :] - image[:, :-1, :, :, :]
    diff_y = image[:, :, 1:, :, :] - image[:, :, :-1, :, :]
    diff_z = image[:, :, :, 1:, :] - image[:, :, :, :-1, :]
    sum_axis = [1, 2, 3, 4]
    tot_var = (tf.reduce_sum(tf.abs(diff_x), axis=sum_axis) + tf.reduce_sum(tf.abs(diff_y), axis=sum_axis) + tf.reduce_sum(tf.abs(diff_z), axis=sum_axis))
    return tf.reduce_sum(tot_var)