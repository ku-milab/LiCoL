import os
import GPUtil

# GPU setting in server
GPU = -1
if GPU == -1: devices = "%d" % GPUtil.getFirstAvailable(order="memory")[0]
else: devices = "%d" % GPU
os.environ["CUDA_VISIBLE_DEVICES"] = devices

########################################
# Mode description  |  Choose the mode #
mode_dict = {"Learn": 0, "Explain": 1}
scenario_dict = {"CN_MCI": 0, "MCI_AD": 1, "CN_AD": 2}
dataset_dict = {"ADNI": 0, "GARD": 1}

mode = "Learn"
scenario = "CN_MCI"
dataset = "ADNI"
########################################

# Data path #
if dataset == "ADNI":
    data_path = "/ADNI/..."
    longitudinal_data_path = "/ADNI/..."
    save_path = "/ADNI/..."
else:
    data_path = "/GARD/..."
    longitudinal_data_path = "/GARD/..."
    save_path = "/GARD/..."

file_name = mode
save_path = save_path + "/%s" % mode

fold = 5
ch = 64
classes = 2

if mode_dict[mode] == 0:
    epoch = 150
    batch_size = 12
    lr, lr_decay = 0.0001, 0.98

elif mode_dict[mode] == 1:
    epoch = 100
    batch_size = 3
    lr_g, lr_d, lr_decay = 0.01, 0.01, 1

    # Weight constants
    cmg_hyper_param = [1.0, 10.0, 10.0, 1.0, 5., 10.0, 5e-6]
    cmg_loss_type = {'cls': cmg_hyper_param[0], 'norm': cmg_hyper_param[1], 'gen': cmg_hyper_param[2], 'cyc': cmg_hyper_param[3],
                     'dis': cmg_hyper_param[4], 'l2': cmg_hyper_param[5], "TV": cmg_hyper_param[6]}

    if mode_dict[mode] == 1:
        if dataset == "ADNI":
            cls_weight_path = "/ADNI/classifier/%dfold_cls_model/variables/variables" % fold
        else:
            cls_weight_path = "/GARD/classifier/%dfold_cls_model/variables/variables" % fold