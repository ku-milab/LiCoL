import CMG_layers as layers
import CMG_config as config
import tensorflow.keras as keras
import tensorflow as tf

class ResNet18:
    def __init__(self, ch=config.ch):
        self.ch = ch
        self.input = layers.input_layer(input_shape=(96, 114, 96, 1), name="input")
        self.build_model()

    def conv_bn_act(self, x, f, n, s=1, k=None, p="same", act=True, trans=False, out_p="auto"):
        if trans:
            c_layer = layers.conv_transpose
        else:
            c_layer = layers.conv

        if k:
            conv_l = c_layer(f=f, p=p, k=k, s=s, out_p=out_p, name=n+"_conv")
        else:
            conv_l = c_layer(f=f, p=p, name=n + "_conv")
        out = conv_l(x)

        norm_l = layers.batch_norm(name=n + "_norm")
        out = norm_l(out)

        if act:
            act_l = layers.relu(name=n + "_relu")
            out = act_l(out)
        return out

    def concat(self, x, y, n):
        concat_l = layers.concat(name=n + "_concat")
        return concat_l([x, y])

    def flatten_layer(self, x, n=None):
        flatten_l = layers.flatten(n + "_flatten")(x)
        return flatten_l

    def dense_layer(self, x, f, act=None, n=None):
        dense_l = layers.dense(f, act=None, name=n + "_dense")
        out = dense_l(x)

        if act:
            act_l = layers.relu(n + "_relu")
            out = act_l(out)
        return out

    def residual_block(self, x, ch, name):
        shortcut = x
        x = self.conv_bn_act(x=x, k=3, f=ch, s=1, p="same", act=True, n="enc_conv%d_1" % name)
        x = self.conv_bn_act(x=x, k=3, f=ch, s=1, p="same", act=False, n="enc_conv%d_2" % name)
        act = layers.relu(name="enc_conv%d_2_relu" % name)
        x = act(x + shortcut)
        return x

    def residual_block_first(self, x, ch, strides, name):
        if x.shape[-1] == ch:
            shortcut = layers.maxpool(k=strides, s=strides, name="max_pool%d" % name)
            shortcut = shortcut(x)
        else:
            shortcut = layers.conv(f=ch, k=1, s=strides, p="same", name="shortcut%d" % name)
            shortcut = shortcut(x)

        x = self.conv_bn_act(x=x, k=3, f=ch, s=strides, p="same", act=True, n="enc_conv%d_1" % name)
        x = self.conv_bn_act(x=x, k=3, f=ch, s=1, p="same", act=False, n="enc_conv%d_2" % name)
        act = layers.relu(name="enc_conv%d_2_relu" % name)
        x = act(x + shortcut)
        return x

    def build_model(self):
        enc_conv1_1 = self.conv_bn_act(x=self.input, f=self.ch, k=7, s=2, p="same", act=True, n="enc_conv1_1")
        max_pool1 = layers.maxpool(k=3, s=2, p="same", name="max_pool1")(enc_conv1_1)

        # conv2_x
        enc_conv2_block1 = self.residual_block(x=max_pool1, ch=self.ch, name=2)
        enc_conv2_block2 = self.residual_block(x=enc_conv2_block1, ch=self.ch, name=3)

        # conv3_x
        enc_conv3_block1 = self.residual_block_first(x=enc_conv2_block2, ch=self.ch*2, strides=2, name=4)
        enc_conv3_block2 = self.residual_block(x=enc_conv3_block1, ch=self.ch*2, name=5)

        # conv4_x
        enc_conv4_block1 = self.residual_block_first(x=enc_conv3_block2, ch=self.ch*4, strides=2, name=6)
        enc_conv4_block2 = self.residual_block(x=enc_conv4_block1, ch=self.ch*4, name=7)

        # conv5_x
        enc_conv5_block1 = self.residual_block_first(x=enc_conv4_block2, ch=self.ch*8, strides=2, name=8)
        enc_conv5_block2 = self.residual_block(x=enc_conv5_block1, ch=self.ch*8, name=9)

        gap = layers.global_avgpool(rank=3, name="gap")(enc_conv5_block2)
        dense = self.dense_layer(x=gap, f=config.classes, act=False, n="dense_2")
        cls_out = layers.softmax(x=dense, name="softmax")

        self.cls_model = keras.Model({"cls_in": self.input}, {"cls_out": cls_out}, name="cls_model")
        return self.cls_model

class ResNet18_Generator:
    def __init__(self, ch=config.ch):
        self.ch = ch
        self.layers = {}
        self.input = layers.input_layer(input_shape=(96, 114, 96, 1), name="input")
        self.c1, self.c2, self.c3, self.c4, self.c5 = layers.resc1(name="c1"), layers.resc2(name="c2"),\
                                             layers.resc3(name="c3"), layers.resc4(name="c4"), layers.resc5(name="c5")
        self.build_model()

    def conv_bn_act(self, x, f, n, s=1, k=None, p="same", act=True, trans=False, out_p="auto"):
        if trans:
            c_layer = layers.conv_transpose
        else:
            c_layer = layers.conv

        if k:
            conv_l = c_layer(f=f, p=p, k=k, s=s, out_p=out_p, name=n + "_conv")
        else:
            conv_l = c_layer(f=f, p=p, name=n + "_conv")
        out = conv_l(x)

        norm_l = layers.batch_norm(name=n + "_norm")
        out = norm_l(out)

        self.layers[n+"_conv"] = conv_l
        self.layers[n+"_norm"] = norm_l

        if act:
            act_l = layers.relu(name=n + "_relu")
            self.layers[n+"_relu"] = act_l
            out = act_l(out)
        return out

    def conv_bn_leakyact(self, x, f, n, s=1, k=None, p="same", act=True, trans=False, out_p="auto"):
        if trans:
            c_layer = layers.conv_transpose
        else:
            c_layer = layers.conv

        if k:
            conv_l = c_layer(f=f, p=p, k=k, s=s, out_p=out_p, name=n + "_conv")
        else:
            conv_l = c_layer(f=f, p=p, name=n + "_conv")
        out = conv_l(x)

        norm_l = layers.batch_norm(name=n + "_norm")
        out = norm_l(out)

        self.layers[n + "_conv"] = conv_l
        self.layers[n + "_norm"] = norm_l

        if act:
            act_l = layers.leaky_relu(name=n + "_leakyrelu")
            self.layers[n + "_leakyrelu"] = act_l
            out = act_l(out)
        return out

    def concat(self, x, y, n):
        concat_l = layers.concat(name=n + "_concat")
        self.layers[n+"_concat"] = concat_l
        return concat_l([x, y])

    def residual_block(self, x, ch, name):
        shortcut = x
        x = self.conv_bn_act(x=x, k=3, f=ch, s=1, p="same", act=True, n="enc_conv%d_1" % name)
        x = self.conv_bn_act(x=x, k=3, f=ch, s=1, p="same", act=False, n="enc_conv%d_2" % name)
        act = layers.relu(name="enc_conv%d_2_relu" % name)
        self.layers["enc_conv%d_2_relu" % name] = act
        x = act(x + shortcut)
        return x

    def residual_block_first(self, x, ch, strides, name):
        if x.shape[-1] == ch:
            shortcut = layers.maxpool(k=strides, s=strides, name="max_pool%d" % name)
            self.layers["max_pool%d" % name] = shortcut
            shortcut = shortcut(x)
        else:
            shortcut = layers.conv(f=ch, k=1, s=strides, p="same", name="shortcut%d" % name)
            self.layers["shortcut%d" % name] = shortcut
            shortcut = shortcut(x)

        x = self.conv_bn_act(x=x, k=3, f=ch, s=strides, p="same", act=True, n="enc_conv%d_1" % name)
        x = self.conv_bn_act(x=x, k=3, f=ch, s=1, p="same", act=False, n="enc_conv%d_2" % name)
        act = layers.relu(name="enc_conv%d_2_relu" % name)
        self.layers["enc_conv%d_2_relu" % name] = act
        x = act(x + shortcut)
        return x

    def build_model(self):
        enc_conv1_1 = self.conv_bn_act(x=self.input, f=self.ch, k=7, s=2, p="same", act=True, n="enc_conv1_1")
        max_pool1 = layers.maxpool(k=3, s=2, p="same", name="max_pool1")(enc_conv1_1)

        # conv2_x
        enc_conv2_block1 = self.residual_block(x=max_pool1, ch=self.ch, name=2)
        enc_conv2_block2 = self.residual_block(x=enc_conv2_block1, ch=self.ch, name=3)

        # conv3_x
        enc_conv3_block1 = self.residual_block_first(x=enc_conv2_block2, ch=self.ch*2, strides=2, name=4)
        enc_conv3_block2 = self.residual_block(x=enc_conv3_block1, ch=self.ch*2, name=5)

        # conv4_x
        enc_conv4_block1 = self.residual_block_first(x=enc_conv3_block2, ch=self.ch*4, strides=2, name=6)
        enc_conv4_block2 = self.residual_block(x=enc_conv4_block1, ch=self.ch*4, name=7)

        # conv5_x
        enc_conv5_block1 = self.residual_block_first(x=enc_conv4_block2, ch=self.ch*8, strides=2, name=8)
        enc_conv5_block2 = self.residual_block(x=enc_conv5_block1, ch=self.ch*8, name=9)

        dec_code_conv5_1 = self.conv_bn_leakyact(x=self.concat(enc_conv5_block2, self.c5, n="dec_code_concat5"), f=self.ch * 8, k=3, s=1, p="same", n="dec_code_conv5_1")
        dec_code_conv5_2 = self.conv_bn_leakyact(x=dec_code_conv5_1, f=self.ch * 8, k=3, s=1, p="same", n="dec_code_conv5_2")

        dec_code_concat4 = self.concat(enc_conv4_block2, self.c4, n="dec_code_concat4_1")
        dec_code_conv4 = self.conv_bn_leakyact(x=dec_code_concat4, f=self.ch * 4, k=3, s=1, p="same", n="dec_code_conv4")
        dec_up4 = layers.upsample(name="dec_up4")(dec_code_conv5_2)
        dec_concat4 = self.concat(dec_up4, dec_code_conv4, n="dec_concat4")
        dec_conv4_1 = self.conv_bn_leakyact(x=dec_concat4, f=self.ch * 4, k=3, s=1, p="same", n="dec_conv4_1")
        dec_conv4_2 = self.conv_bn_leakyact(x=self.concat(dec_conv4_1, dec_code_conv4, n="dec_code_concat4_2"), f=self.ch * 4, k=3, s=1, p="same", n="dec_conv4_2")

        dec_code_concat3 = self.concat(enc_conv3_block2, self.c3, n="dec_code_concat3_1")
        dec_code_conv3 = self.conv_bn_leakyact(x=dec_code_concat3, f=self.ch * 2, k=3, s=1, p="same", n="dec_code_conv3")
        dec_up3 = layers.upsample(name="dec_up3")(dec_conv4_2)
        dec_up3 = self.conv_bn_act(x=dec_up3, f=self.ch * 2, k=(1, 2, 1), s=1, p="valid", n="dec_up3_conv")
        dec_concat3 = self.concat(dec_up3, dec_code_conv3, n="dec_concat3")
        dec_conv3_1 = self.conv_bn_leakyact(x=dec_concat3, f=self.ch * 2, k=3, s=1, p="same", n="dec_conv3_1")
        dec_conv3_2 = self.conv_bn_leakyact(x=self.concat(dec_conv3_1, dec_code_conv3, n="dec_code_concat3_2"), f=self.ch * 2, k=3, s=1, p="same", n="dec_conv3_2")

        dec_code_concat2 = self.concat(enc_conv2_block2, self.c2, n="dec_code_concat2_1")
        dec_code_conv2 = self.conv_bn_leakyact(x=dec_code_concat2, f=self.ch, k=3, s=1, p="same", n="dec_code_conv2")
        dec_up2 = layers.upsample(name="dec_up2")(dec_conv3_2)
        dec_up2 = self.conv_bn_act(x=dec_up2, f=self.ch, k=(1, 2, 1), s=1, p="valid", n="dec_up2_conv")
        dec_concat2 = self.concat(dec_up2, dec_code_conv2, n="dec_concat2")
        dec_conv2_1 = self.conv_bn_leakyact(x=dec_concat2, f=self.ch, k=3, s=1, p="same", n="dec_conv2_1")
        dec_conv2_2 = self.conv_bn_leakyact(x=self.concat(dec_conv2_1, dec_code_conv2, n="dec_code_concat2_2"), f=self.ch, k=3, s=1, p="same", n="dec_conv2_2")

        dec_code_concat1 = self.concat(enc_conv1_1, self.c1, n="dec_code_concat1_1")
        dec_code_conv1 = self.conv_bn_leakyact(x=dec_code_concat1, f=self.ch, k=3, s=1, p="same", n="dec_code_conv1")
        dec_up1 = layers.upsample(name="dec_up1")(dec_conv2_2)
        dec_up1 = self.conv_bn_act(x=dec_up1, f=self.ch, k=(1, 2, 1), s=1, p="valid", n="dec_up1_conv")
        dec_concat1 = self.concat(dec_up1, dec_code_conv1, n="dec_concat1")
        dec_conv1_1 = self.conv_bn_leakyact(x=dec_concat1, f=self.ch, k=3, s=1, p="same", n="dec_conv1_1")
        dec_conv1_2 = self.conv_bn_leakyact(x=self.concat(dec_conv1_1, dec_code_conv1, n="dec_code_concat1_2"), f=self.ch, k=3, s=1, p="same", n="dec_conv1_2")

        # Only upsampling
        dec_up = layers.upsample(name="dec_up")(dec_conv1_2)
        dec_conv = self.conv_bn_leakyact(x=dec_up, f=1, k=3, s=1, p="same", act=False, n="dec_conv")
        dec_out = tf.identity(dec_conv)

        self.dec_model = keras.Model({"gen_in": self.input, "c1": self.c1, "c2": self.c2, "c3": self.c3, "c4": self.c4, "c5": self.c5},
                                     {"gen_out": dec_out}, name="dec_model")
        return self.dec_model

class ResNet18_Discriminator:
    def __init__(self, ch=config.ch):
        self.ch = ch
        self.discri_in_layer = layers.input_layer(input_shape=(96, 114, 96, 1), name="input")
        self.build_model()

    def conv_bn_act(self, x, f, n, s=1, k=None, p="same", act=True, trans=False, out_p="auto"):
        if trans:
            c_layer = layers.conv_transpose
        else:
            c_layer = layers.conv

        if k:
            conv_l = c_layer(f=f, p=p, k=k, s=s, out_p=out_p, name=n+"_conv")
        else:
            conv_l = c_layer(f=f, p=p, name=n + "_conv")
        out = conv_l(x)

        norm_l = layers.batch_norm(name=n + "_norm")
        out = norm_l(out)

        if act:
            act_l = layers.relu(name=n + "_relu")
            out = act_l(out)
        return out

    def concat(self, x, y, n):
        concat_l = layers.concat(name=n + "_concat")
        return concat_l([x, y])

    def flatten_layer(self, x, n=None):
        flatten_l = layers.flatten(n + "_flatten")(x)
        return flatten_l

    def dense_layer(self, x, f, act=None, n=None):
        dense_l = layers.dense(f, act=None, name=n + "_dense")
        out = dense_l(x)

        if act:
            act_l = layers.relu(n + "_relu")
            out = act_l(out)
        return out

    def residual_block(self, x, ch, name):
        shortcut = x
        x = self.conv_bn_act(x=x, k=3, f=ch, s=1, p="same", act=True, n="enc_conv%d_1" % name)
        x = self.conv_bn_act(x=x, k=3, f=ch, s=1, p="same", act=False, n="enc_conv%d_2" % name)
        act = layers.relu(name="enc_conv%d_2_relu" % name)
        x = act(x + shortcut)
        return x

    def residual_block_first(self, x, ch, strides, name):
        if x.shape[-1] == ch:
            shortcut = layers.maxpool(k=strides, s=strides, name="max_pool%d" % name)
            shortcut = shortcut(x)
        else:
            shortcut = layers.conv(f=ch, k=1, s=strides, p="same", name="shortcut%d" % name)
            shortcut = shortcut(x)

        x = self.conv_bn_act(x=x, k=3, f=ch, s=strides, p="same", act=True, n="enc_conv%d_1" % name)
        x = self.conv_bn_act(x=x, k=3, f=ch, s=1, p="same", act=False, n="enc_conv%d_2" % name)
        act = layers.relu(name="enc_conv%d_2_relu" % name)
        x = act(x + shortcut)
        return x

    def build_model(self):
        enc_conv1_1 = self.conv_bn_act(x=self.discri_in_layer, f=self.ch, k=7, s=2, p="same", act=True, n="enc_conv1_1")
        max_pool1 = layers.maxpool(k=3, s=2, p="same", name="max_pool1")(enc_conv1_1)

        # conv2_x
        enc_conv2_block1 = self.residual_block(x=max_pool1, ch=self.ch, name=2)
        enc_conv2_block2 = self.residual_block(x=enc_conv2_block1, ch=self.ch, name=3)

        # conv3_x
        enc_conv3_block1 = self.residual_block_first(x=enc_conv2_block2, ch=self.ch*2, strides=2, name=4)
        enc_conv3_block2 = self.residual_block(x=enc_conv3_block1, ch=self.ch*2, name=5)

        # conv4_x
        enc_conv4_block1 = self.residual_block_first(x=enc_conv3_block2, ch=self.ch*4, strides=2, name=6)
        enc_conv4_block2 = self.residual_block(x=enc_conv4_block1, ch=self.ch*4, name=7)

        # conv5_x
        enc_conv5_block1 = self.residual_block_first(x=enc_conv4_block2, ch=self.ch*8, strides=2, name=8)
        enc_conv5_block2 = self.residual_block(x=enc_conv5_block1, ch=self.ch*8, name=9)

        gap = layers.global_avgpool(rank=3, name="gap")(enc_conv5_block2)
        fc = self.dense_layer(x=gap, f=1, act=False, n="fc")
        out = tf.identity(input=fc, name="identity")

        self.discriminator_model = keras.Model({"discri_in": self.discri_in_layer}, {"discri_out": out}, name="dis_model")
        return self.discriminator_model