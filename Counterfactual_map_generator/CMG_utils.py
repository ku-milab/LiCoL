import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
import CMG_config as config

class Utils:
    def __init__(self):
        self.data_path = config.data_path
        self.long_path = config.longitudinal_data_path
        self.padding = 5

    def load_adni_data(self):
        """
        class #0: NC(433), class #1: pMCI(251), class #2: sMCI(497), class #3: AD(359)
        :return: NC, pMCI, sMCI, AD
        """
        dat = np.load(self.data_path + "total_dat.npy", mmap_mode="r")
        lbl = np.load(self.data_path + "labels.npy")
        return dat, lbl

    def load_gard_data(self):
        """
        class #0: CN(261), class #1: MCI(375), class #3: AD(109)
        :return: (CN, MCI, AD) == 745
        """
        dat = np.load(self.data_path + "total_avg_dat.npy", mmap_mode="r")
        lbl = np.load(self.data_path + "total_lbl.npy")
        return dat, lbl

    def load_longitudinal_data(self):
        long_nc, long_mci, long_ad = np.load(self.long_path+"/resized_quan_NC.npy"), np.load(self.long_path+"/resized_quan_MCI.npy"), np.load(self.long_path+"/resized_quan_AD.npy")
        long_nc, long_mci, long_ad = np.expand_dims(long_nc, axis=-1), np.expand_dims(long_mci, axis=-1), np.expand_dims(long_ad, axis=-1)
        return long_nc, long_mci, long_ad

    def adni_data_permutation(self, lbl, cv):
        if config.scenario == "CN_AD":
            Total_z_idx, Total_o_idx = np.squeeze(np.argwhere(lbl == 0)), np.squeeze(np.argwhere(lbl == 3))
        elif config.scenario == "CN_MCI":
            Total_z_idx, Total_o_idx = np.squeeze(np.argwhere(lbl == 0)), np.append(np.squeeze(np.argwhere(lbl == 1)), np.squeeze(np.argwhere(lbl == 2)))
        elif config.scenario == "MCI_AD":
            Total_z_idx, Total_o_idx = np.append(np.squeeze(np.argwhere(lbl == 1)), np.squeeze(np.argwhere(lbl == 2))), np.squeeze(np.argwhere(lbl == 3))

        amount_z, amount_o = len(Total_z_idx) // 5, len(Total_o_idx) // 5

        Zvalid_idx = Total_z_idx[cv * amount_z:(cv + 1) * amount_z]
        Ztrain_idx = np.setdiff1d(Total_z_idx, Zvalid_idx)
        Ztest_idx = Zvalid_idx[:int(len(Zvalid_idx) / 2)]
        Zvalid_idx = np.setdiff1d(Zvalid_idx, Ztest_idx)

        Ovalid_idx = Total_o_idx[cv * amount_o:(cv + 1) * amount_o]
        Otrain_idx = np.setdiff1d(Total_o_idx, Ovalid_idx)
        Otest_idx = Ovalid_idx[:int(len(Ovalid_idx) / 2)]
        Ovalid_idx = np.setdiff1d(Ovalid_idx, Otest_idx)

        train_all_idx = np.concatenate((Ztrain_idx, Otrain_idx))
        valid_all_idx = np.concatenate((Zvalid_idx, Ovalid_idx))
        test_all_idx = np.concatenate((Ztest_idx, Otest_idx))

        return train_all_idx, valid_all_idx, test_all_idx

    def adni_separate_data(self, data_idx, dat, lbl, CENTER=False):
        dat, lbl = dat[data_idx], lbl[data_idx]
        dat = np.squeeze(dat)
        if config.scenario == "CN_AD":
            lbl = np.where(lbl == 3, 1, lbl).astype("int32")
        elif config.scenario == "CN_MCI":
            lbl = np.where(lbl == 2, 1, lbl).astype("int32")
        elif config.scenario == "MCI_AD":
            lbl = np.where(lbl == 1, 0, lbl).astype("int32")
            lbl = np.where(lbl == 2, 0, lbl).astype("int32")
            lbl = np.where(lbl == 3, 1, lbl).astype("int32")

        lbl = np.eye(2)[lbl.squeeze()]
        lbl = lbl.astype('float32')
        if len(data_idx) > config.batch_size: dat = np.expand_dims(dat, axis=0)

        # Original
        if CENTER:
            for batch in range(len(data_idx)):
                # Quantile normalization
                Q1, Q3 = np.quantile(dat[batch], 0.1), np.quantile(dat[batch], 0.9)
                dat[batch] = np.where(dat[batch] < Q1, Q1, dat[batch])
                dat[batch] = np.where(dat[batch] > Q3, Q3, dat[batch])

                # Gaussian normalization
                m, std = np.mean(dat[batch]), np.std(dat[batch])
                dat[batch] = (dat[batch] - m) / std
            dat = np.expand_dims(dat, axis=-1)

        else:
            padding = 5
            npad = ((padding, padding), (padding, padding), (padding, padding))
            emp = np.empty(shape=(dat.shape[0], dat.shape[1], dat.shape[2], dat.shape[3]))

            for cnt, dat in enumerate(dat):
                tmp = np.pad(dat, npad, "constant")
                emp[cnt] = tf.image.random_crop(tmp, emp[cnt].shape)

            for batch in range(len(emp)):
                # Quantile normalization
                Q1, Q3 = np.quantile(emp[batch], 0.1), np.quantile(emp[batch], 0.9)
                emp[batch] = np.where(emp[batch] < Q1, Q1, emp[batch])
                emp[batch] = np.where(emp[batch] > Q3, Q3, emp[batch])

                # Gaussian normalization
                m, std = np.mean(emp[batch]), np.std(emp[batch])
                emp[batch] = (emp[batch] - m) / std
            dat = np.expand_dims(emp, axis=-1)

        return dat, lbl

    def gard_data_permutation(self, lbl, cv):
        if config.scenario == "CN_AD":
            Total_z_idx, Total_o_idx = np.squeeze(np.argwhere(lbl == 0)), np.squeeze(np.argwhere(lbl == 2))
        elif config.scenario == "CN_MCI":
            Total_z_idx, Total_o_idx = np.squeeze(np.argwhere(lbl == 0)), np.squeeze(np.argwhere(lbl == 1))
        elif config.scenario == "MCI_AD":
            Total_z_idx, Total_o_idx = np.squeeze(np.argwhere(lbl == 1)), np.squeeze(np.argwhere(lbl == 2))

        amount_z, amount_o = len(Total_z_idx) // 5, len(Total_o_idx) // 5

        Zvalid_idx = Total_z_idx[cv * amount_z:(cv + 1) * amount_z]
        Ztrain_idx = np.setdiff1d(Total_z_idx, Zvalid_idx)
        Ztest_idx = Zvalid_idx[:int(len(Zvalid_idx) / 2)]
        Zvalid_idx = np.setdiff1d(Zvalid_idx, Ztest_idx)

        Ovalid_idx = Total_o_idx[cv * amount_o:(cv + 1) * amount_o]
        Otrain_idx = np.setdiff1d(Total_o_idx, Ovalid_idx)
        Otest_idx = Ovalid_idx[:int(len(Ovalid_idx) / 2)]
        Ovalid_idx = np.setdiff1d(Ovalid_idx, Otest_idx)

        train_all_idx = np.concatenate((Ztrain_idx, Otrain_idx))
        valid_all_idx = np.concatenate((Zvalid_idx, Ovalid_idx))
        test_all_idx = np.concatenate((Ztest_idx, Otest_idx))

        return train_all_idx, valid_all_idx, test_all_idx

    def gard_separate_data(self, data_idx, dat, lbl, CENTER=False):
        dat, lbl = dat[data_idx], lbl[data_idx]
        dat = np.squeeze(dat)
        if config.scenario == "CN_AD":
            lbl = np.where(lbl == 2, 1, lbl).astype("int32")
        elif config.scenario == "CN_MCI":
            lbl = lbl.astype("int32")
        elif config.scenario == "MCI_AD":
            lbl = np.where(lbl == 1, 0, lbl).astype("int32")
            lbl = np.where(lbl == 2, 1, lbl).astype("int32")

        lbl = np.eye(2)[lbl.squeeze()]
        lbl = lbl.astype('float32')
        if len(data_idx) > config.batch_size: dat = np.expand_dims(dat, axis=0)

        # Original
        if CENTER:
            for batch in range(len(data_idx)):
                # Quantile normalization
                Q1, Q3 = np.quantile(dat[batch], 0.1), np.quantile(dat[batch], 0.9)
                dat[batch] = np.where(dat[batch] < Q1, Q1, dat[batch])
                dat[batch] = np.where(dat[batch] > Q3, Q3, dat[batch])

                # Gaussian normalization
                m, std = np.mean(dat[batch]), np.std(dat[batch])
                dat[batch] = (dat[batch] - m) / std
            dat = np.expand_dims(dat, axis=-1)

        else:
            padding = 5
            npad = ((padding, padding), (padding, padding), (padding, padding))
            emp = np.empty(shape=(dat.shape[0], dat.shape[1], dat.shape[2], dat.shape[3]))

            for cnt, dat in enumerate(dat):
                tmp = np.pad(dat, npad, "constant")
                emp[cnt] = tf.image.random_crop(tmp, emp[cnt].shape)

            for batch in range(len(emp)):
                # Quantile normalization
                Q1, Q3 = np.quantile(emp[batch], 0.1), np.quantile(emp[batch], 0.9)
                emp[batch] = np.where(emp[batch] < Q1, Q1, emp[batch])
                emp[batch] = np.where(emp[batch] > Q3, Q3, emp[batch])

                # Gaussian normalization
                m, std = np.mean(emp[batch]), np.std(emp[batch])
                emp[batch] = (emp[batch] - m) / std
            dat = np.expand_dims(emp, axis=-1)

        return dat, lbl

    def code_creator(self, train_dat_size):
        # Original
        for i in range(0, train_dat_size):
            code = np.random.choice(config.classes, config.classes, replace=False).astype("float32")
            code = np.expand_dims(code, axis=0)
            if i == 0:
                target_c = code
            else:
                target_c = np.append(target_c, code, axis=0)
        target_c = target_c[:train_dat_size]
        target_c = np.where(target_c == 2., 0., target_c)
        return target_c

    def codemap(self, condition):
        c1, c2 = np.zeros((len(condition), 48, 57, 48, config.classes)), np.zeros((len(condition), 24, 29, 24, config.classes))
        c3, c4 = np.zeros((len(condition), 12, 15, 12, config.classes)), np.zeros((len(condition), 6, 8, 6, config.classes))
        c5 = np.zeros((len(condition), 3, 4, 3, config.classes))
        for batch in range(len(condition)):
            for classes in range(condition.shape[-1]):
                c1[batch, ..., classes], c2[batch, ..., classes] = condition[batch, classes], condition[batch, classes]
                c3[batch, ..., classes], c4[batch, ..., classes] = condition[batch, classes], condition[batch, classes]
                c5[batch, ..., classes] = condition[batch, classes]
        return c1, c2, c3, c4, c5