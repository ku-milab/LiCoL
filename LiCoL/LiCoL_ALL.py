import os
import LiCoL_layers as layers
import GPUtil
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn import metrics
import tqdm
import nibabel as nib
import time

GPU = -1

if GPU == -1:
    devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
else:
    devices = "%d" % GPU

os.environ["CUDA_VISIBLE_DEVICES"] = devices
tf.keras.backend.set_image_data_format("channels_last")
l = keras.layers
K = keras.backend

######### Hyper-parameter setting #########
fold = 5
epoch = 100
learning_rate = 0.0001
learning_decay = 0.98
batch_size = 6
ch = 256

dataset = "ADNI"
scenario = "CN_MCI"
pseudo_name = "GM_density_map_classification"
###########################################

if scenario == "CN_AD":
    if dataset == "ADNI":
        query_size, key_value_size = 79, 79
    elif dataset == "GARD":
        query_size, key_value_size = 86, 86

elif scenario == "CN_MCI":
    if dataset == "ADNI":
        query_size, key_value_size = 56, 56
    elif dataset == "GARD":
        query_size, key_value_size = 81, 81

elif scenario == "MCI_AD":
    if dataset == "ADNI":
        query_size, key_value_size = 75, 75
    elif dataset == "GARD":
        query_size, key_value_size = 80, 80

def aal3_description():
    aal_dict = {}
    aal_dict[1] = 'Precentral', 'L.PreCG'
    aal_dict[2] = 'Precentral', 'R.PreCG'
    aal_dict[3] = 'Frontal_Sup', 'L.SFG'
    aal_dict[4] = 'Frontal_Sup', 'R.SFG'
    aal_dict[5] = 'Frontal_Mid', 'L.MFG'
    aal_dict[6] = 'Frontal_Mid', 'R.MFG'
    aal_dict[7] = 'Frontal_Inf_Oper', 'L.IFGoperc'
    aal_dict[8] = 'Frontal_Inf_Oper', 'R.IFGoperc'
    aal_dict[9] = 'Frontal_Inf_Tri', 'L.IFGtriang'
    aal_dict[10] = 'Frontal_Inf_Tri', 'R.IFGtriang'
    aal_dict[11] = 'Frontal_Inf_Orb', 'L.IFGorb'
    aal_dict[12] = 'Frontal_Inf_Orb', 'R.IFGorb'
    aal_dict[13] = 'Rolandic_Oper', 'L.ROL'
    aal_dict[14] = 'Rolandic_Oper', 'R.ROL'
    aal_dict[15] = 'Supp_Motor_Area', 'L.SMA'
    aal_dict[16] = 'Supp_Motor_Area', 'R.SMA'
    aal_dict[17] = 'Olfactory', 'L.OLF'
    aal_dict[18] = 'Olfactory', 'R.OLF'
    aal_dict[19] = 'Frontal_Sup_Med', 'L.SFGmedial'
    aal_dict[20] = 'Frontal_Sup_Med', 'R.SFGmedial'
    aal_dict[21] = 'Frontal_Med_Orb', 'L.PFCventmed'
    aal_dict[22] = 'Frontal_Med_Orb', 'R.PFCventmed'
    aal_dict[23] = 'Rectus', 'L.REC'
    aal_dict[24] = 'Rectus', 'R.REC'
    aal_dict[25] = 'OFCmed', 'L.OFCmed'
    aal_dict[26] = 'OFCmed', 'R.OFCmed'
    aal_dict[27] = 'OFCant', 'L.OFCant'
    aal_dict[28] = 'OFCant', 'R.OFCant'
    aal_dict[29] = 'OFCpost', 'L.OFCpost'
    aal_dict[30] = 'OFCpost', 'R.OFCpost'
    aal_dict[31] = 'OFClat', 'L.OFClat'
    aal_dict[32] = 'OFClat', 'R.OFClat'
    aal_dict[33] = 'Insula', 'L.INS'
    aal_dict[34] = 'Insula', 'R.INS'
    aal_dict[35] = 'Cingulate_Ant', 'L.ACC'
    aal_dict[36] = 'Cingulate_Ant', 'R.ACC'
    aal_dict[37] = 'Cingulate_Mid', 'L.MCC'
    aal_dict[38] = 'Cingulate_Mid', 'R.MCC'
    aal_dict[39] = 'Cingulate_Post', 'L.PCC'
    aal_dict[40] = 'Cingulate_Post', 'R.PCC'
    aal_dict[41] = 'Hippocampus', 'L.HIP'
    aal_dict[42] = 'Hippocampus', 'R.HIP'
    aal_dict[43] = 'ParaHippocampal', 'L.PHG'
    aal_dict[44] = 'ParaHippocampal', 'R.PHG'
    aal_dict[45] = 'Amygdala', 'L.AMYG'
    aal_dict[46] = 'Amygdala', 'R.AMYG'
    aal_dict[47] = 'Calcarine', 'L.CAL'
    aal_dict[48] = 'Calcarine', 'R.CAL'
    aal_dict[49] = 'Cuneus', 'L.CUN'
    aal_dict[50] = 'Cuneus', 'R.CUN'
    aal_dict[51] = 'Lingual', 'L.LING'
    aal_dict[52] = 'Lingual', 'R.LING'
    aal_dict[53] = 'Occipital_Sup', 'L.SOG'
    aal_dict[54] = 'Occipital_Sup', 'R.SOG'
    aal_dict[55] = 'Occipital_Mid', 'L.MOG'
    aal_dict[56] = 'Occipital_Mid', 'R.MOG'
    aal_dict[57] = 'Occipital_Inf', 'L.IOG'
    aal_dict[58] = 'Occipital_Inf', 'R.IOG'
    aal_dict[59] = 'Fusiform', 'L.FFG'
    aal_dict[60] = 'Fusiform', 'R.FFG'
    aal_dict[61] = 'Postcentral', 'L.PoCG'
    aal_dict[62] = 'Postcentral', 'R.PoCG'
    aal_dict[63] = 'Parietal_Sup', 'L.SPG'
    aal_dict[64] = 'Parietal_Sup', 'R.SPG'
    aal_dict[65] = 'Parietal_Inf', 'L.IPG'
    aal_dict[66] = 'Parietal_Inf', 'R.IPG'
    aal_dict[67] = 'SupraMarginal', 'L.SMG'
    aal_dict[68] = 'SupraMarginal', 'R.SMG'
    aal_dict[69] = 'Angular', 'L.ANG'
    aal_dict[70] = 'Angular', 'R.ANG'
    aal_dict[71] = 'Precuneus', 'L.PCUN'
    aal_dict[72] = 'Precuneus', 'R.PCUN'
    aal_dict[73] = 'Paracentral_Lobule', 'L.PCL'
    aal_dict[74] = 'Paracentral_Lobule', 'R.PCL'
    aal_dict[75] = 'Caudate', 'L.CAU'
    aal_dict[76] = 'Caudate', 'R.CAU'
    aal_dict[77] = 'Putamen', 'L.PUT'
    aal_dict[78] = 'Putamen', 'R.PUT'
    aal_dict[79] = 'Pallidum', 'L.PAL'
    aal_dict[80] = 'Pallidum', 'R.PAL'
    aal_dict[81] = 'Thalamus', 'L.THA'
    aal_dict[82] = 'Thalamus', 'R.THA'
    aal_dict[83] = 'Heschl', 'L.HES'
    aal_dict[84] = 'Heschl', 'R.HES'
    aal_dict[85] = 'Temporal_Sup', 'L.STG'
    aal_dict[86] = 'Temporal_Sup', 'R.STG'
    aal_dict[87] = 'Temporal_Pole_Sup', 'L.TPOsup'
    aal_dict[88] = 'Temporal_Pole_Sup', 'R.TPOsup'
    aal_dict[89] = 'Temporal_Mid', 'L.MTG'
    aal_dict[90] = 'Temporal_Mid', 'R.MTG'
    aal_dict[91] = 'Temporal_Pole_Mid', 'L.TPOmid'
    aal_dict[92] = 'Temporal_Pole_Mid', 'R.TPOmid'
    aal_dict[93] = 'Temporal_Inf', 'L.ITG'
    aal_dict[94] = 'Temporal_Inf', 'R.ITG'
    aal_dict[95] = 'Cerebellum_Crus1', 'L.CERCRU1'
    aal_dict[96] = 'Cerebellum_Crus1', 'R.CERCRU1'
    aal_dict[97] = 'Cerebellum_Crus2', 'L.CERCRU2'
    aal_dict[98] = 'Cerebellum_Crus2', 'R.CERCRU2'
    aal_dict[99] = 'Cerebellum_3', 'L.CER3'
    aal_dict[100] = 'Cerebellum_3', 'R.CER3'
    aal_dict[101] = 'Cerebellum_4_5', 'L.CER4_5'
    aal_dict[102] = 'Cerebellum_4_5', 'R.CER4_5'
    aal_dict[103] = 'Cerebellum_6', 'L.CER6'
    aal_dict[104] = 'Cerebellum_6', 'R.CER6'
    aal_dict[105] = 'Cerebellum_7b', 'L.CER7b'
    aal_dict[106] = 'Cerebellum_7b', 'R.CER7b'
    aal_dict[107] = 'Cerebellum_8', 'L.CER8'
    aal_dict[108] = 'Cerebellum_8', 'R.CER8'
    aal_dict[109] = 'Cerebellum_9', 'L.CER9'
    aal_dict[110] = 'Cerebellum_9', 'R.CER9'
    aal_dict[111] = 'Cerebellum_10', 'L.CER10'
    aal_dict[112] = 'Cerebellum_10', 'R.CER10'
    aal_dict[113] = 'Vermis_1_2', 'VER1_2'
    aal_dict[114] = 'Vermis_3', 'VER3'
    aal_dict[115] = 'Vermis_4_5', 'VER4_5'
    aal_dict[116] = 'Vermis_6', 'VER6'
    aal_dict[117] = 'Vermis_7', 'VER7'
    aal_dict[118] = 'Vermis_8', 'VER8'
    aal_dict[119] = 'Vermis_9', 'VER9'
    aal_dict[120] = 'Vermis_10', 'VER10'
    aal_dict[121] = 'Thal_AV', 'L.tAV'
    aal_dict[122] = 'Thal_AV', 'R.tAV'
    aal_dict[123] = 'Thal_LP', 'L.tLP'
    aal_dict[124] = 'Thal_LP', 'R.tLP'
    aal_dict[125] = 'Thal_VA', 'L.tVA'
    aal_dict[126] = 'Thal_VA', 'R.tVA'
    aal_dict[127] = 'Thal_VL', 'L.tVL'
    aal_dict[128] = 'Thal_VL', 'R.tVL'
    aal_dict[129] = 'Thal_VPL', 'L.tVPL'
    aal_dict[130] = 'Thal_VPL', 'R.tVPL'
    aal_dict[131] = 'Thal_IL', 'L.tIL'
    aal_dict[132] = 'Thal_IL', 'R.tIL'
    aal_dict[133] = 'Thal_Re', 'L.tRe'
    aal_dict[134] = 'Thal_Re', 'R.tRe'
    aal_dict[135] = 'Thal_MDm', 'L.tMDm'
    aal_dict[136] = 'Thal_MDm', 'R.tMDm'
    aal_dict[137] = 'Thal_MDl', 'L.tMDl'
    aal_dict[138] = 'Thal_MDl', 'R.tMDl'
    aal_dict[139] = 'Thal_LGN', 'L.tLGN'
    aal_dict[140] = 'Thal_LGN', 'R.tLGN'
    aal_dict[141] = 'Thal_MGN', 'L.tMGN'
    aal_dict[142] = 'Thal_MGN', 'R.tMGN'
    aal_dict[143] = 'Thal_PuA', 'L.tPuA'
    aal_dict[144] = 'Thal_PuA', 'R.tPuA'
    aal_dict[145] = 'Thal_PuM', 'L.tPuM'
    aal_dict[146] = 'Thal_PuM', 'R.tPuM'
    aal_dict[147] = 'Thal_PuL', 'L.tPuL'
    aal_dict[148] = 'Thal_PuL', 'R.tPuL'
    aal_dict[149] = 'Thal_PuI', 'L.tPuI'
    aal_dict[150] = 'Thal_PuI', 'R.tPuI'
    aal_dict[151] = 'ACC_sub', 'L.ACCsub'
    aal_dict[152] = 'ACC_sub', 'R.ACCsub'
    aal_dict[153] = 'ACC_pre', 'L.ACCpre'
    aal_dict[154] = 'ACC_pre', 'R.ACCpre'
    aal_dict[155] = 'ACC_sup', 'L.ACCsup'
    aal_dict[156] = 'ACC_sup', 'R.ACCsup'
    aal_dict[157] = 'N_Acc', 'L.Nacc'
    aal_dict[158] = 'N_Acc', 'R.Nacc'
    aal_dict[159] = 'VTA', 'L.VTA'
    aal_dict[160] = 'VTA', 'R.VTA'
    aal_dict[161] = 'SN_pc', 'L.SNpc'
    aal_dict[162] = 'SN_pc', 'R.SNpc'
    aal_dict[163] = 'SN_pr', 'L.SNpr'
    aal_dict[164] = 'SN_pr', 'R.SNpr'
    aal_dict[165] = 'Red_N', 'L.RedN'
    aal_dict[166] = 'Red_N', 'R.RedN'
    aal_dict[167] = 'LC', 'L.LC'
    aal_dict[168] = 'LC', 'R.LC'
    aal_dict[169] = 'Raphe_D', 'RapheD'
    aal_dict[170] = 'Raphe_M', 'RapheM'
    return aal_dict

class Utils:
    def __init__(self, scenario, network):
        self.scenario = scenario
        self.network = network
        self.density_type = "GM_mod_merg_s2.nii.gz"
        self.data_path = "/DataCommon2/ksoh/classification_performance/map_nonLinear_registration/densitymap/%s" % dataset
        self.w, self.h, self.d = 91, 109, 91

    def load_data(self):
        dat = nib.load(os.path.join(self.data_path, self.network, self.scenario, self.density_type))
        lbl = nib.load(os.path.join(self.data_path, self.network, self.scenario, "labels.npy")).astype("int32")
        dat = np.array(dat.dataobj)

        ddat = np.empty((dat.shape[-1], self.w, self.h, self.d))
        for i in tqdm.trange(len(ddat), desc="Data re-indexing"): ddat[i] = dat[..., i]

        gt_zero_idx, gt_one_idx = np.squeeze(np.argwhere(lbl == 0)), np.squeeze(np.argwhere(lbl == 1))
        syn_zero_idx, syn_one_idx = np.squeeze(np.argwhere(lbl == 2)), np.squeeze(np.argwhere(lbl == 3))

        gt_zero_dat, gt_one_dat = ddat[gt_zero_idx], ddat[gt_one_idx]
        gt_zero_lbl, gt_one_lbl = np.zeros(len(gt_zero_dat)), np.ones(len(gt_one_dat))
        syn_zero_dat, syn_one_dat = ddat[syn_zero_idx], ddat[syn_one_idx]
        syn_zero_lbl, syn_one_lbl = np.zeros(len(syn_zero_dat)), np.ones(len(syn_one_dat))

        return np.expand_dims(np.concatenate([gt_zero_dat, gt_one_dat], axis=0), axis=-1), np.append(gt_zero_lbl, gt_one_lbl),\
               np.expand_dims(np.concatenate([syn_zero_dat, syn_one_dat], axis=0), axis=-1), np.append(syn_zero_lbl, syn_one_lbl)

    def evaluation_matrics(self, y_true, y_pred):
        acc = K.mean(K.equal(y_true, np.round(y_pred)))
        auc = metrics.roc_auc_score(y_true, y_pred)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, np.round(y_pred)).ravel()
        sen = tp / (tp + fn)
        spe = tn / (fp + tn)
        return acc, auc, sen, spe

class LiCoL:
    def __init__(self, ch=ch, query_size=query_size, key_value_size=query_size):
        self.ch = ch
        self.q, self.k_v = query_size, key_value_size
        self.query = layers.input_layer(input_shape=(self.q, 1), name="input")
        self.key_value = layers.input_layer(input_shape=(self.k_v, 1), name="CCA_input")
        self.build_model()

    def build_model(self):
        q_embed = layers.conv(f=self.ch, k=1, s=1, p="same", rank=1, dilation_rate=1)(self.query)
        k_embed = layers.conv(f=self.ch, k=1, s=1, p="same", rank=1, dilation_rate=1)(self.key_value)
        v_embed = layers.conv(f=self.ch, k=1, s=1, p="same", rank=1, dilation_rate=1)(self.key_value)
        d_k = k_embed.get_shape().as_list()[1]

        attention = tf.matmul(q_embed, tf.transpose(k_embed, [0, 2, 1]))
        attention /= d_k ** 0.5
        attention = layers.softmax(attention, axis=-1)

        out = tf.matmul(attention, v_embed)

        # CCA_ROI_cls_Mode1
        out = tf.add(tf.reduce_mean(self.query, axis=[-1]), tf.reduce_mean(out, axis=[-1]))
        out = layers.sigmoid(tf.reduce_mean(out, axis=[-1]))

        self.cls_model = keras.Model({"q_in": self.query, "k_v_in": self.key_value},
                                     {"attention": attention, "cls_out": out}, name="cls_model")
        return self.cls_model

class Trainer:
    def __init__(self):
        self.lr = learning_rate
        self.decay = learning_decay
        self.epoch = epoch
        self.batch_size = batch_size

        self.valid_acc, self.compare_acc, self.count = 0, 0, 0
        tf.keras.backend.set_image_data_format("channels_last")
        self.valid_save, self.nii_save, self.model_select = False, False, False
        self.build_model()

        self.path = "/DataCommon2/ksoh/classification_performance/map_nonLinear_registration/Classification/%s/%s" % (scenario, pseudo_name)
        if not os.path.exists(self.path): os.makedirs(self.path)

    def build_model(self):
        model = LiCoL()
        self.train_vars = []
        self.cls_model = model.build_model()
        self.train_vars += self.cls_model.trainable_variables

    def _train_one_batch(self, query, key_value, lbl, gen_optim, train_vars):
        with tf.GradientTape() as tape:
            res = self.cls_model({"q_in": query, "k_v_in": key_value}, training=True)["cls_out"]
            loss = K.mean(keras.losses.binary_crossentropy(lbl, res))

        grads = tape.gradient(loss, train_vars)
        gen_optim.apply_gradients(zip(grads, train_vars))
        return loss

    def train(self):
        util = Utils(scenario, "resnet")
        gt_dat, gt_lbl, syn_dat, syn_lbl = util.load_data()
        self.build_model()

        for i in tqdm.trange(len(gt_dat), desc="GT Data re-indexing with Gaussian norm"):
            m, std = np.mean(gt_dat[i]), np.std(gt_dat[i])
            gt_dat[i] = (gt_dat[i] - m) / std

        for i in tqdm.trange(len(syn_dat), desc="SYN Data re-indexing with Gaussian norm"):
            m, std = np.mean(syn_dat[i]), np.std(syn_dat[i])
            syn_dat[i] = (syn_dat[i] - m) / std

        for cv in range(fold):
            Total_z_idx, Total_o_idx = len(np.argwhere(gt_lbl == 0)[0]), len(np.where(gt_lbl == 1)[0])
            amount_z, amount_o = len(Total_z_idx) // 5, len(Total_o_idx) // 5

            Zvalid_idx = Total_z_idx[cv * amount_z:(cv + 1) * amount_z]
            Ztrain_idx = np.setdiff1d(Total_z_idx, Zvalid_idx)
            Ztest_idx = Zvalid_idx[:int(len(Zvalid_idx) / 2)]
            Zvalid_idx = np.setdiff1d(Zvalid_idx, Ztest_idx)

            Ovalid_idx = Total_o_idx[cv * amount_o:(cv + 1) * amount_o]
            Otrain_idx = np.setdiff1d(Total_o_idx, Ovalid_idx)
            Otest_idx = Ovalid_idx[:int(len(Ovalid_idx) / 2)]
            Ovalid_idx = np.setdiff1d(Ovalid_idx, Otest_idx)

            trn_all_idx = np.concatenate((Ztrain_idx, Otrain_idx))
            val_all_idx = np.concatenate((Zvalid_idx, Ovalid_idx))
            tst_all_idx = np.concatenate((Ztest_idx, Otest_idx))

            trn_dat, val_dat, tst_dat = gt_dat[trn_all_idx], gt_dat[val_all_idx], gt_dat[tst_all_idx]
            trn_lbl, val_lbl, tst_lbl = gt_lbl[trn_all_idx], gt_lbl[val_all_idx], gt_lbl[tst_all_idx]

            # Data augmentation using counterfactual-labeled sMRI data
            trn_dat, trn_lbl = np.concatenate((trn_dat, syn_dat), axis=0), np.append(trn_lbl, syn_lbl)

            self.train_summary_writer = tf.summary.create_file_writer(self.path + "/%dfold_train" % (cv + 1))
            self.valid_summary_writer = tf.summary.create_file_writer(self.path + "/%dfold_valid" % (cv + 1))
            self.test_summary_writer = tf.summary.create_file_writer(self.path + "/%dfold_test" % (cv + 1))

            lr_schedule = keras.optimizers.schedules.ExponentialDecay(self.lr, decay_steps=len(trn_dat) // self.batch_size, decay_rate=self.decay, staircase=True)
            optim = keras.optimizers.Adam(lr_schedule)

            if util.scenario == "CN_AD":
                CCA_mean = nib.load("/DataCommon2/ksoh/classification_performance/map_nonLinear_registration/densitymap/%s/resnet/%d_results/CCA/CN_AD_diff_mean.nii.gz" % (dataset, cv))
                CCA_binary = nib.load("/DataCommon2/ksoh/classification_performance/map_nonLinear_registration/densitymap/%s/resnet/%d_results/CCA/CN_AD_diff_CCA_99.nii.gz" % (dataset, cv))
            elif util.scenario == "CN_MCI":
                CCA_mean = nib.load("/DataCommon2/ksoh/classification_performance/map_nonLinear_registration/densitymap/%s/resnet/%d_results/CCA/CN_MCI_diff_mean.nii.gz" % (dataset, cv))
                CCA_binary = nib.load("/DataCommon2/ksoh/classification_performance/map_nonLinear_registration/densitymap/%s/resnet/%d_results/CCA/CN_MCI_diff_CCA_99.nii.gz" % (dataset, cv))
            elif util.scenario == "MCI_AD":
                CCA_mean = nib.load("/DataCommon2/ksoh/classification_performance/map_nonLinear_registration/densitymap/%s/resnet/%d_results/CCA/MCI_AD_diff_mean.nii.gz" % (dataset, cv))
                CCA_binary = nib.load("/DataCommon2/ksoh/classification_performance/map_nonLinear_registration/densitymap/%s/resnet/%d_results/CCA/MCI_AD_diff_CCA_99.nii.gz" % (dataset, cv))
            else: print("Scenario Error !!!")

            CCA_mean = np.array(CCA_mean.dataobj).squeeze()
            CCA_binary = np.array(CCA_binary.dataobj).squeeze()

            aal3 = nib.load("/DataCommon2/ksoh/classification_performance/map_nonLinear_registration/densitymap/AAL3v1.nii.gz")
            aal3 = np.array(aal3.dataobj)
            diff_temp = CCA_binary * aal3

            uni_roi = np.delete(np.unique(diff_temp), 0)

            query_vector = np.empty((len(uni_roi)))
            for cnt, roi in enumerate(uni_roi):
                query_vector[cnt] = CCA_mean[np.where(diff_temp == roi)].mean()
            assert len(uni_roi) == len(query_vector)

            for cur_epoch in tqdm.trange(self.epoch, desc="Classification - %s" % (pseudo_name)):
                idx = np.random.permutation(np.arange(len(trn_dat)))

                for cur_step in tqdm.trange(0, len(trn_dat), self.batch_size):   # Training
                    cur_dat, cur_lbl = trn_dat[idx[cur_step:cur_step+self.batch_size]], trn_lbl[idx[cur_step:cur_step+self.batch_size]]
                    cur_dat = np.squeeze(cur_dat)
                    if len(cur_dat) > self.batch_size: cur_dat = np.expand_dims(cur_dat, axis=0)

                    query, selected_trn_feat = np.empty((len(cur_dat), query_size)), np.empty((len(cur_dat), key_value_size))
                    for cnt1, dat in enumerate(cur_dat):
                        query[cnt1] = query_vector
                        for cnt2, roi in enumerate(uni_roi):
                            selected_trn_feat[cnt1][cnt2] = dat[np.where(diff_temp == roi)].mean()

                    query, selected_trn_feat = np.expand_dims(query, axis=-1), np.expand_dims(selected_trn_feat, axis=-1)
                    trn_loss = self._train_one_batch(query=query, key_value=selected_trn_feat, lbl=cur_lbl, gen_optim=optim, train_vars=self.train_vars)

                    if cur_step % 10 == 0:
                        with self.train_summary_writer.as_default():
                            tf.summary.scalar("%dfold_train_loss" % (cv + 1), trn_loss, step=cur_step)

                # Validation
                tot_true, tot_pred = 0, 0
                idx = np.arange(len(val_dat))
                for val_step in tqdm.trange(0, len(val_dat), self.batch_size, desc="Validation"):
                    cur_dat, cur_lbl = val_dat[idx[val_step:val_step + self.batch_size]], val_lbl[idx[val_step:val_step + self.batch_size]]
                    if len(cur_dat) > batch_size: cur_dat = np.expand_dims(cur_dat, axis=0)

                    query, selected_val_feat = np.empty((len(cur_dat), len(uni_roi))), np.empty((len(cur_dat), len(uni_roi)))
                    for cnt1, dat in enumerate(cur_dat):
                        query[cnt1] = query_vector
                        for cnt2, roi in enumerate(uni_roi):
                            selected_val_feat[cnt1, cnt2] = dat[np.where(diff_temp == roi)].mean()

                    query, selected_val_feat = np.expand_dims(query, axis=-1), np.expand_dims(selected_val_feat, axis=-1)
                    res = self.cls_model({"q_in": query, "k_v_in": selected_val_feat}, training=False)["cls_out"]

                    if tst_step == 0: tot_pred = res
                    else: tot_pred = np.append(tot_pred, res)

                acc, auc, sen, spe = util.evaluation_matrics(tot_true, tot_pred)

                if self.compare_acc <= acc:
                    self.cls_model.save(self.path + "/cls_model_%03d" % (cur_epoch))
                    self.compare_acc = acc

                with self.test_summary_writer.as_default():
                    tf.summary.scalar("%dfold_validation_ACC" % (cv + 1), acc, step=cur_epoch)
                    tf.summary.scalar("%dfold_validation_AUC" % (cv + 1), auc, step=cur_epoch)
                    tf.summary.scalar("%dfold_validation_SEN" % (cv + 1), sen, step=cur_epoch)
                    tf.summary.scalar("%dfold_validation_SPE" % (cv + 1), spe, step=cur_epoch)

                # Test
                tot_true, tot_pred = 0, 0
                idx = np.arange(len(tst_dat))
                for tst_step in tqdm.trange(0, len(tst_dat), self.batch_size, desc="Testing"):
                    cur_dat, cur_lbl = tst_dat[idx[tst_step:tst_step + self.batch_size]], tst_lbl[idx[tst_step:tst_step + self.batch_size]]
                    if len(cur_dat) > batch_size: cur_dat = np.expand_dims(cur_dat, axis=0)

                    query, selected_tst_feat = np.empty((len(cur_dat), len(uni_roi))), np.empty((len(cur_dat), len(uni_roi)))
                    for cnt1, dat in enumerate(cur_dat):
                        query[cnt1] = query_vector
                        for cnt2, roi in enumerate(uni_roi):
                            selected_tst_feat[cnt1, cnt2] = dat[np.where(diff_temp == roi)].mean()

                    query, selected_tst_feat = np.expand_dims(query, axis=-1), np.expand_dims(selected_tst_feat, axis=-1)
                    res = self.cls_model({"q_in": query, "k_v_in": selected_tst_feat}, training=False)["cls_out"]

                    if tst_step == 0: tot_pred = res
                    else: tot_pred = np.append(tot_pred, res)

                acc, auc, sen, spe = util.evaluation_matrics(tot_true, tot_pred)
                with self.test_summary_writer.as_default():
                    tf.summary.scalar("%dfold_test_ACC" % (cv + 1), acc, step=cur_epoch)
                    tf.summary.scalar("%dfold_test_AUC" % (cv + 1), auc, step=cur_epoch)
                    tf.summary.scalar("%dfold_test_SEN" % (cv + 1), sen, step=cur_epoch)
                    tf.summary.scalar("%dfold_test_SPE" % (cv + 1), spe, step=cur_epoch)

tr=Trainer()
tr.train()