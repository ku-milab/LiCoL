import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import CMG_config as config
import CMG_networks as network
import CMG_losses as loss
import CMG_test as test
import CMG_utils as utils
import tqdm

l = keras.layers
K = keras.backend

class Trainer:
    def __init__(self):
        self.file_name = config.file_name
        self.fold = config.fold

        self.valid_acc, self.compare_acc, self.valid_loss, self.compare_loss, self.count = 0, 0, 0, 100, 0
        tf.keras.backend.set_image_data_format("channels_last")
        self.valid_save, self.nii_save, self.model_select = False, False, False
        self.build_model()

        self.path = config.save_path
        if not os.path.exists(self.path): os.makedirs(self.path)

    def build_model(self):
        if config.mode == "Learn":
            resnet = network.ResNet18()
            self.train_vars = []
            self.cls_model = resnet.build_model()
            self.train_vars += self.cls_model.trainable_variables

        elif config.mode == "Explain":
            generator, discriminator = network.ResNet18_Generator(), network.ResNet18_Discriminator()
            self.train_discri_vars = []

            self.discri_model = discriminator.build_model()
            self.train_discri_vars += self.discri_model.trainable_variables
            self.gen_model = generator.build_model()

            resnet = network.ResNet18()
            self.train_vars = []
            self.cls_model = resnet.build_model()

            cls_load_weights = config.cls_weight_path
            self.cls_model.load_weights(cls_load_weights)
            for layer in self.cls_model.layers: layer.trainable = False

            for enc_layer, gen_layer in zip(self.cls_model.layers[:-3], self.gen_model.layers):
                gen_layer.set_weights(enc_layer.get_weights())
                gen_layer.trainable = False

            save_variables = False
            for variables in self.gen_model.trainable_variables:
                if "dec" in variables.name: save_variables = True
                if save_variables: self.train_vars += [variables]

    def _train_one_batch(self, dat_all, lbl, gen_optim, train_vars, step, cv):
        with tf.GradientTape() as tape:
            res = self.cls_model({"cls_in": dat_all}, training=True)
            train_loss = loss.BCE_loss(lbl, res["cls_out"])

        grads = tape.gradient(train_loss, train_vars)
        gen_optim.apply_gradients(zip(grads, train_vars))

        if step % 10 == 0:
            with self.train_summary_writer.as_default():
                tf.summary.scalar("%dfold_train_loss" % (cv + 1), train_loss, step=step)

    def cycle_consistency(self, pseudo_image, source):
        c1, c2, c3, c4, c5 = utils.Utils().codemap(condition=source)
        tilde_map = self.gen_model({"gen_in": pseudo_image, "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5}, training=True)["gen_out"]
        return pseudo_image + tilde_map

    def _GAN_train_one_batch(self, dat_all, gen_optim, disc_optim, target_c, train_vars, train_discri_vars, step):
        label = self.cls_model({"cls_in": dat_all}, training=False)["cls_out"]
        t1, t2, t3, t4, t5 = utils.Utils().codemap(condition=target_c)

        # Discriminator step
        with tf.GradientTape() as tape:
            cfmap = self.gen_model({"gen_in": dat_all, "c1": t1, "c2": t2, "c3": t3, "c4": t4, "c5": t5}, training=True)["gen_out"]
            pseudo_image = dat_all + cfmap
            dat_all_like = self.cycle_consistency(pseudo_image, label)

            real = self.discri_model({"discri_in": dat_all}, training=True)["discri_out"]
            real_like = self.discri_model({"discri_in": dat_all_like}, training=True)["discri_out"]
            fake = self.discri_model({"discri_in": pseudo_image}, training=True)["discri_out"]

            real_loss = (loss.MSE_loss(tf.ones_like(real), real) + loss.MSE_loss(tf.ones_like(real_like), real_like))/2
            fake_loss = loss.MSE_loss(tf.zeros_like(fake), fake)
            train_dis_loss = config.cmg_loss_type["dis"] * (real_loss + fake_loss)

        grads = tape.gradient(train_dis_loss, train_discri_vars)
        disc_optim.apply_gradients(zip(grads, train_discri_vars))

        if step % 10 == 0:
            with self.discriminator_summary_writer.as_default():
                tf.summary.scalar("discriminator_loss", train_dis_loss, step=step)

        # Generator step
        with tf.GradientTape() as tape:
            cfmap = self.gen_model({"gen_in": dat_all, "c1": t1, "c2": t2, "c3": t3, "c4": t4, "c5": t5}, training=True)["gen_out"]
            pseudo_image = dat_all + cfmap
            dat_all_like = self.cycle_consistency(pseudo_image, label)

            fake = self.discri_model({"discri_in": pseudo_image}, training=True)["discri_out"]
            fake_like = self.discri_model({"discri_in": dat_all_like}, training=True)["discri_out"]

            pred = self.cls_model({"cls_in": pseudo_image}, training=False)["cls_out"]

            cyc_loss = loss.cycle_loss(dat_all, dat_all_like)
            l1_loss, l2_loss = loss.L1_norm(effect_map=cfmap), loss.L2_norm(effect_map=cfmap)
            gen_loss = (loss.MSE_loss(tf.ones_like(fake), fake) + loss.MSE_loss(tf.ones_like(fake_like), fake_like))/2
            cls_loss = loss.BCE_loss(target_c, pred)
            tv_loss = loss.tv_loss(pseudo_image)

            cyc = config.cmg_loss_type["cyc"] * cyc_loss
            l1 = config.cmg_loss_type["norm"] * l1_loss
            l2 = config.cmg_loss_type["l2"] * l2_loss
            gen = config.cmg_loss_type["gen"] * gen_loss
            cls = config.cmg_loss_type["cls"] * cls_loss
            tv = config.cmg_loss_type["TV"] * tv_loss
            train_gen_loss = cyc + l1 + l2 + gen + cls + tv

        grads = tape.gradient(train_gen_loss, train_vars)
        gen_optim.apply_gradients(zip(grads, train_vars))

        if step % 10 == 0:
            with self.generator_summary_writer.as_default():
                tf.summary.scalar("mode1_G_total_train_loss", train_gen_loss, step=step)
                tf.summary.scalar("generator_cyc_loss", cyc, step=step)
                tf.summary.scalar("generator_l1_loss", l1, step=step)
                tf.summary.scalar("generator_l2_loss", l2, step=step)
                tf.summary.scalar("generator_gen_loss", gen, step=step)
                tf.summary.scalar("generator_cls_loss", cls, step=step)
                tf.summary.scalar("generator_tv_loss", tv, step=step)

    def _valid_logger(self, dat_all, lbl, epoch, cv):
        res = self.cls_model({"cls_in": dat_all}, training=False)["cls_out"]
        valid_loss = loss.BCE_loss(lbl, res)
        val_acc, val_auc, val_sen, val_spe = test.evaluation_matrics(lbl, res)

        self.valid_loss += valid_loss
        self.valid_acc += val_acc
        self.valid_auc += val_auc
        self.valid_sen += val_sen
        self.valid_spe += val_spe
        self.count += 1

        if self.valid_save == True:
            self.valid_acc = self.valid_acc / self.count
            self.valid_auc = self.valid_auc / self.count
            self.valid_sen = self.valid_sen / self.count
            self.valid_spe = self.valid_spe / self.count
            self.valid_loss = self.valid_loss / self.count

            if self.compare_acc <= self.valid_acc:
                self.model_select = True
                self.compare_acc = self.valid_acc

            with self.valid_summary_writer.as_default():
                tf.summary.scalar("%dfold_valid_loss" % (cv + 1), self.valid_loss, step=epoch)
                tf.summary.scalar("%dfold_valid_acc" % (cv + 1), self.valid_acc, step=epoch)
                tf.summary.scalar("%dfold_valid_auc" % (cv + 1), self.valid_auc, step=epoch)
                tf.summary.scalar("%dfold_valid_sen" % (cv + 1), self.valid_sen, step=epoch)
                tf.summary.scalar("%dfold_valid_spe" % (cv + 1), self.valid_spe, step=epoch)
                self.valid_acc, self.valid_auc, self.valid_sen, self.valid_spe, self.valid_loss, self.count = 0, 0, 0, 0, 0, 0
                self.valid_save = False

    def _GAN_valid_logger(self, dat_all, target_c, epoch):
        label = self.cls_model({"cls_in": dat_all}, training=False)["cls_out"]
        t1, t2, t3, t4, t5 = utils.Utils().codemap(condition=target_c)

        cfmap = self.gen_model({"gen_in": dat_all, "c1": t1, "c2": t2, "c3": t3, "c4": t4, "c5": t5}, training=True)["gen_out"]
        pseudo_image = dat_all + cfmap
        dat_all_like = self.cycle_consistency(pseudo_image, label)

        fake = self.discri_model({"discri_in": pseudo_image}, training=True)["discri_out"]
        fake_like = self.discri_model({"discri_in": dat_all_like}, training=True)["discri_out"]

        pred = self.cls_model({"cls_in": pseudo_image}, training=False)["cls_out"]

        cyc_loss = loss.cycle_loss(dat_all, dat_all_like)
        l1_loss, l2_loss = loss.L1_norm(effect_map=cfmap), loss.L2_norm(effect_map=cfmap)
        gen_loss = (loss.MSE_loss(tf.ones_like(fake), fake) + loss.MSE_loss(tf.ones_like(fake_like), fake_like)) / 2
        cls_loss = loss.BCE_loss(target_c, pred)
        tv_loss = loss.tv_loss(pseudo_image)

        cyc = config.cmg_loss_type["cyc"] * cyc_loss
        l1 = config.cmg_loss_type["norm"] * l1_loss
        l2 = config.cmg_loss_type["l2"] * l2_loss
        gen = config.cmg_loss_type["gen"] * gen_loss
        cls = config.cmg_loss_type["cls"] * cls_loss
        tv = config.cmg_loss_type["TV"] * tv_loss

        self.valid_loss += cyc + l1 + l2 + gen + cls + tv
        self.count += 1

        if self.valid_save == True:
            self.valid_loss = self.valid_loss / self.count

            if self.valid_loss <= self.compare_loss:
                self.model_select = True
                self.compare_loss = self.valid_loss

            with self.valid_summary_writer.as_default():
                tf.summary.scalar("valid_loss", self.valid_loss, step=epoch)
                self.valid_loss, self.count = 0, 0
                self.valid_save = False

    def cls_train(self):
        if config.dataset == "ADNI":
            dat, lbl = utils.Utils().load_adni_data()
        elif config.dataset == "GARD":
            dat, lbl = utils.Utils().load_gard_data()
        else: print("Data load error !!")

        for cv in range(0, self.fold):
            self.train_summary_writer = tf.summary.create_file_writer(self.path + "/%dfold_train" % (cv + 1))
            self.valid_summary_writer = tf.summary.create_file_writer(self.path + "/%dfold_valid" % (cv + 1))
            self.test_summary_writer = tf.summary.create_file_writer(self.path + "/%dfold_test" % (cv + 1))

            if config.dataset == "ADNI":
                self.train_all_idx, self.valid_all_idx, self.test_all_idx = utils.Utils().adni_data_permutation(lbl, cv)
            elif config.dataset == "GARD":
                self.train_all_idx, self.valid_all_idx, self.test_all_idx = utils.Utils().gard_data_permutation(lbl, cv)
            self.build_model()

            lr_schedule = keras.optimizers.schedules.ExponentialDecay(config.lr, decay_steps=len(self.train_all_idx) // config.batch_size, decay_rate=config.lr_decay, staircase=True)
            optim = keras.optimizers.Adam(lr_schedule)
            global_step, self.compare_acc = 0, 0

            for cur_epoch in tqdm.trange(config.epoch, desc="Resnet18_%s.py" % self.file_name):
                self.train_all_idx = np.random.permutation(self.train_all_idx)

                # training
                for cur_step in tqdm.trange(0, len(self.train_all_idx), config.batch_size, desc="%dfold_%depoch_%s" % (cv + 1, cur_epoch, self.file_name)):
                    cur_idx = self.train_all_idx[cur_step:cur_step + config.batch_size]

                    if config.dataset == "ADNI":
                        cur_dat, cur_lbl = utils.Utils().adni_separate_data(cur_idx, dat, lbl, CENTER=False)
                    elif config.dataset == "GARD":
                        cur_dat, cur_lbl = utils.Utils().gard_separate_data(cur_idx, dat, lbl, CENTER=False)

                    self._train_one_batch(dat_all=cur_dat, lbl=cur_lbl, gen_optim=optim, train_vars=self.train_vars, step=global_step, cv=cv)
                    global_step += 1

                # validation
                for val_step in tqdm.trange(0, len(self.valid_all_idx), config.batch_size, desc="Validation step: %dfold" % (cv + 1)):
                    val_idx = self.valid_all_idx[val_step:val_step + config.batch_size]

                    if config.dataset == "ADNI":
                        val_dat, val_lbl = utils.Utils().adni_separate_data(val_idx, dat, lbl, CENTER=True)
                    elif config.dataset == "GARD":
                        val_dat, val_lbl = utils.Utils().gard_separate_data(val_idx, dat, lbl, CENTER=True)

                    if val_step + config.batch_size >= len(self.valid_all_idx): self.valid_save = True
                    self._valid_logger(dat_all=val_dat, lbl=val_lbl, epoch=cur_epoch, cv=cv)

                if self.model_select == True:
                    self.cls_model.save(os.path.join(self.path + '/%dfold_cls_model' % (cv + 1)))
                    self.model_select = False

                # Test
                tot_true, tot_pred = 0, 0
                for tst_step in tqdm.trange(0, len(self.test_all_idx), config.batch_size, desc="Testing step: %dfold" % (cv + 1)):
                    tst_idx = self.test_all_idx[tst_step:tst_step + config.batch_size]

                    if config.dataset == "ADNI":
                        tst_dat, tst_lbl = utils.Utils().adni_separate_data(tst_idx, dat, lbl, CENTER=True)
                    elif config.dataset == "GARD":
                        tst_dat, tst_lbl = utils.Utils().gard_separate_data(tst_idx, dat, lbl, CENTER=True)

                    res = self.cls_model({"cls_in": tst_dat}, training=False)["cls_out"]

                    if tst_step == 0: tot_true, tot_pred = tst_lbl, res
                    else:
                        tot_true, tot_pred = np.concatenate((tot_true, tst_lbl), axis=0), np.concatenate((tot_pred, res), axis=0)

                acc, auc, sen, spe = test.evaluation_matrics(tot_true, tot_pred)
                with self.test_summary_writer.as_default():
                    tf.summary.scalar("%dfold_test_ACC" % (cv + 1), acc, step=cur_epoch)
                    tf.summary.scalar("%dfold_test_AUC" % (cv + 1), auc, step=cur_epoch)
                    tf.summary.scalar("%dfold_test_SEN" % (cv + 1), sen, step=cur_epoch)
                    tf.summary.scalar("%dfold_test_SPE" % (cv + 1), spe, step=cur_epoch)

    def gan_train(self):
        if config.dataset == "ADNI":
            dat, lbl = utils.Utils().load_adni_data()
        elif config.dataset == "GARD":
            dat, lbl = utils.Utils().load_gard_data()
        else: print("Data load error !!")

        for cv in range(0, self.fold):
            self.discriminator_summary_writer = tf.summary.create_file_writer(self.path + "/%dfold_train_critic" % (cv + 1))
            self.generator_summary_writer = tf.summary.create_file_writer(self.path + "/%dfold_train_generator" % (cv + 1))
            self.valid_summary_writer = tf.summary.create_file_writer(self.path + "/%dfold_valid_generator" % (cv + 1))

            if config.dataset == "ADNI":
                self.train_all_idx, self.valid_all_idx, self.test_all_idx = utils.Utils().adni_data_permutation(lbl, cv)
            elif config.dataset == "GARD":
                self.train_all_idx, self.valid_all_idx, self.test_all_idx = utils.Utils().gard_data_permutation(lbl, cv)
            self.build_model()

            g_lr_schedule = keras.optimizers.schedules.ExponentialDecay(config.lr_g, decay_steps=len(self.train_all_idx) // config.batch_size, decay_rate=config.lr_decay, staircase=True)
            d_lr_schedule = keras.optimizers.schedules.ExponentialDecay(config.lr_d, decay_steps=len(self.train_all_idx) // config.batch_size, decay_rate=config.lr_decay, staircase=True)
            gen_optim = keras.optimizers.Adam(learning_rate=g_lr_schedule)
            disc_optim = keras.optimizers.Adam(learning_rate=d_lr_schedule)
            global_step = 0

            f = open(self.path + "/%fold_all_ncc_result.txt" % (cv+1), "w")
            f.write("|  Zero-like ACC  |  One-like ACC  |  NCC(+)  |  NCC(-)  |\n")
            f.close()

            for cur_epoch in tqdm.trange(config.epoch, desc=config.file_name):
                self.train_all_idx = np.random.permutation(self.train_all_idx)

                # training
                for cur_step in tqdm.trange(0, len(self.train_all_idx), config.batch_size, desc="%dfold_%depoch_%s" % (cv + 1, cur_epoch, self.file_name)):
                    cur_idx = self.train_all_idx[cur_step:cur_step + config.batch_size]

                    if config.dataset == "ADNI":
                        cur_dat, cur_lbl = utils.Utils().adni_separate_data(cur_idx, dat, lbl, CENTER=False)
                    elif config.dataset == "GARD":
                        cur_dat, cur_lbl = utils.Utils().gard_separate_data(cur_idx, dat, lbl, CENTER=False)
                    target_idx = utils.Utils().code_creator(len(cur_idx))

                    self._GAN_train_one_batch(dat_all=cur_dat, gen_optim=gen_optim, disc_optim=disc_optim, target_c=target_idx,
                                              train_vars=self.train_vars, train_discri_vars=self.train_discri_vars, step=global_step)
                    global_step += 1

                # validation
                for val_step in tqdm.trange(0, len(self.valid_all_idx), config.batch_size, desc="Validation step: %dfold" % (cv + 1)):
                    val_idx = self.valid_all_idx[val_step:val_step + config.batch_size]

                    if config.dataset == "ADNI":
                        val_dat, val_lbl = utils.Utils().adni_separate_data(val_idx, dat, lbl, CENTER=True)
                    elif config.dataset == "GARD":
                        val_dat, val_lbl = utils.Utils().gard_separate_data(val_idx, dat, lbl, CENTER=True)

                    target_idx = utils.Utils().code_creator(len(val_idx))

                    if val_step + config.batch_size >= len(self.valid_all_idx): self.valid_save = True
                    self._GAN_valid_logger(dat_all=val_dat, target_c=target_idx, epoch=cur_epoch)

                if self.model_select == True:
                    self.gen_model.save(os.path.join(self.path + '/%dfold_gen_model' % (cv + 1)))
                    self.model_select = False

                # Test
                ncc_p, ncc_n, z_like_acc, o_like_acc = test.ncc_evaluation(self.cls_model, self.gen_model)
                f = open(self.path + "/%fold_all_ncc_result.txt" % (cv+1), "a")
                f.write("Epoch:%03d -> z_like_acc:%.3f | o_like_acc: %.3f | ncc_p:%.3f | ncc_n: %.3f |\n" % (cur_epoch, z_like_acc, o_like_acc, ncc_p, ncc_n))
                f.close()

Tr = Trainer()
if config.mode == "Learn": Tr.cls_train()
elif config.mode == "Explain": Tr.gan_train()